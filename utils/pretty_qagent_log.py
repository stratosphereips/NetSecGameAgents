#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rich summary viewer for QAgent logs.

Default mode (summary) groups low-level log lines into high-level ACTION STEPS:
  [timestamp] Step N: <ActionType.XYZ>  (reward Δ, new discoveries)

Each step panel includes:
  • Concept vs Real action (mapped hosts / networks)
  • Parameters table
  • Environment response (reward, end flag)
  • Counts & diffs (new hosts, services, networks, data)
  • Raw JSON (optional toggle)

Raw mode (--mode raw) reproduces the original line-by-line colored view.

Color piping: colors are forced when stdout is not a TTY so you can do: | less -R

Examples:
  python utils/pretty_qagent_log.py path/to/q_agent.log                 # summary
  python utils/pretty_qagent_log.py --mode raw path/to/q_agent.log      # raw view
  python utils/pretty_qagent_log.py --limit 20 path/to/q_agent.log      # first 20 steps
  python utils/pretty_qagent_log.py --search FindServices path/log      # filter steps containing substring
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Any

from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.traceback import install

install(show_locals=False)

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
LINE_RE = re.compile(r"^(?P<ts>\d{2}:\d{2}:\d{2})\s+(?P<component>\S+)\s+(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+(?P<msg>.*)$")
CONCEPT_ACTION_RE = re.compile(r"\[\+\]\s+Concept Action selected:Action <([^>]+)>")
REAL_ACTION_RE = re.compile(r"\[\+\]\s+Real Action selected:Action <([^>]+)>")
REWARD_RE = re.compile(r"\[\+\]\s+Reward of last action.*: ([-+]?\d+)")
SENDING_PREFIX = "Sending: "
RECEIVED_PREFIX = "Data received from env: "

# ---------------------------------------------------------------------------
@dataclass
class Step:
    index: int
    timestamp: str
    concept: Optional[str] = None
    real: Optional[str] = None
    sending: Optional[dict] = None
    received: Optional[dict] = None
    reward: Optional[int] = None
    end: Optional[bool] = None  # whether this step ended the episode (from env)
    raw_lines: List[str] = field(default_factory=list)
    # Parsed textual state when structured JSON is absent (INFO-level logs)
    parsed_state: Optional[dict] = None  # shape similar to normalize_state output

    # Diffs (computed later)
    new_hosts: List[str] = field(default_factory=list)
    new_networks: List[str] = field(default_factory=list)
    new_services: Dict[str, List[str]] = field(default_factory=dict)
    new_data: Dict[str, List[str]] = field(default_factory=dict)

    # Episode bookkeeping (assigned post-parse)
    episode: int = 1
    step_in_episode: int = 0


END_FLAG_RE = re.compile(r"end=(True|False)")  # in 'State after action:' lines

# Regexes for INFO-level textual state dumps (I2C lines)
REAL_STATE_NETS_RE = re.compile(r"I2C: Real state known nets: \{([^}]*)\}")
REAL_STATE_HOSTS_RE = re.compile(r"I2C: Real state known hosts: \{([^}]*)\}")
REAL_STATE_CONTROLLED_RE = re.compile(r"I2C: Real state controlled hosts: \{([^}]*)\}")
REAL_STATE_SERVICES_RE = re.compile(r"I2C: Real state known services: \{([^}]*)\}")
REAL_STATE_DATA_RE = re.compile(r"I2C: Real state known data: \{([^}]*)\}")

# Service name extraction inside service listing
SERVICE_NAME_RE = re.compile(r"Service\(name='([^']+)'")


def parse_log_lines(path: Path) -> List[Step]:
    """Parse log into structured steps.

    NOTE: Concept / Real action lines are NOT preceded by timestamps in the raw log.
    We therefore keep the last seen timestamp from any preceding line and assign it
    to subsequent action lines until a new timestamped line is encountered.
    """
    steps: List[Step] = []
    current: Optional[Step] = None
    last_ts: str = "??:??:??"

    for raw_line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        m = LINE_RE.match(raw_line)
        if m:
            # Timestamped line
            last_ts = m.group('ts')
            msg = m.group('msg')
            # Attach raw line to current step (for context) AFTER we possibly detect step boundaries
            concept_m = CONCEPT_ACTION_RE.search(msg)
            if concept_m:
                current = Step(index=len(steps)+1, timestamp=last_ts, concept=concept_m.group(1).strip())
                steps.append(current)
                continue

            if current is not None:
                current.raw_lines.append(raw_line)
                # Attempt parsing of textual state fragments on timestamped lines too
                _maybe_parse_textual_state(current, msg)

            # Within a step, parse attributes
            if current is not None:
                real_m = REAL_ACTION_RE.search(msg)
                if real_m:
                    current.real = real_m.group(1).strip()
                    continue

                reward_m = REWARD_RE.search(msg)
                if reward_m:
                    try:
                        current.reward = int(reward_m.group(1))
                    except ValueError:
                        pass
                    continue

                if msg.startswith(SENDING_PREFIX):
                    payload = msg[len(SENDING_PREFIX):].strip()
                    try:
                        current.sending = json.loads(payload)
                    except Exception:
                        pass
                    continue

                if msg.startswith(RECEIVED_PREFIX):
                    payload = msg[len(RECEIVED_PREFIX):].strip()
                    try:
                        current.received = json.loads(payload)
                        # Extract end flag if available in JSON structure
                        try:
                            recv_obj = current.received if isinstance(current.received, dict) else {}
                            obs = recv_obj.get('observation', {}) or {}
                            if isinstance(obs, dict):
                                end_flag = obs.get('end')
                                if isinstance(end_flag, bool):
                                    current.end = end_flag
                        except Exception:
                            pass
                    except Exception:
                        pass
                    continue
        else:
            # Non-timestamp line (possible concept/real/reward lines)
            concept_m = CONCEPT_ACTION_RE.search(raw_line)
            if concept_m:
                current = Step(index=len(steps)+1, timestamp=last_ts, concept=concept_m.group(1).strip())
                steps.append(current)
                continue
            if current is None:
                continue
            # Add raw line to current context
            current.raw_lines.append(raw_line)
            # Parse textual state lines
            _maybe_parse_textual_state(current, raw_line)
            real_m = REAL_ACTION_RE.search(raw_line)
            if real_m:
                current.real = real_m.group(1).strip()
                continue
            reward_m = REWARD_RE.search(raw_line)
            if reward_m:
                try:
                    current.reward = int(reward_m.group(1))
                except ValueError:
                    pass
                continue
            if raw_line.startswith(SENDING_PREFIX):
                payload = raw_line[len(SENDING_PREFIX):].strip()
                try:
                    current.sending = json.loads(payload)
                except Exception:
                    pass
                continue
            if raw_line.startswith(RECEIVED_PREFIX):
                payload = raw_line[len(RECEIVED_PREFIX):].strip()
                try:
                    current.received = json.loads(payload)
                    try:
                        recv_obj = current.received if isinstance(current.received, dict) else {}
                        obs = recv_obj.get('observation', {}) or {}
                        if isinstance(obs, dict):
                            end_flag = obs.get('end')
                            if isinstance(end_flag, bool):
                                current.end = end_flag
                    except Exception:
                        pass
                except Exception:
                    pass
                continue
            # Fallback textual detection of end flag in 'State after action' representation
            m_end = END_FLAG_RE.search(raw_line)
            if m_end:
                current.end = (m_end.group(1) == 'True')
    return steps


def _maybe_parse_textual_state(step: Step, line: str):
    """Populate step.parsed_state from INFO-level textual 'I2C: Real state ...' lines.

    Only initializes data structure when a pattern actually matches to avoid
    creating empty states that could interfere with diffing heuristics.
    """
    matched = False
    nets_m = REAL_STATE_NETS_RE.search(line)
    hosts_m = REAL_STATE_HOSTS_RE.search(line)
    controlled_m = REAL_STATE_CONTROLLED_RE.search(line)
    services_m = REAL_STATE_SERVICES_RE.search(line)
    data_m = REAL_STATE_DATA_RE.search(line)
    if not any([nets_m, hosts_m, controlled_m, services_m, data_m]):
        return
    # Now initialize container since something matched
    if step.parsed_state is None:
        step.parsed_state = {
            'networks': set(),
            'hosts': set(),
            'controlled': set(),
            'services': {},
            'data': {},
        }
    ps = step.parsed_state
    if nets_m:
        nets_raw = [n.strip() for n in nets_m.group(1).split(',') if n.strip()]
        ps['networks'].update(nets_raw)
    if hosts_m:
        hosts_raw = [h.strip() for h in hosts_m.group(1).split(',') if h.strip()]
        ps['hosts'].update(hosts_raw)
    if controlled_m:
        controlled_raw = [h.strip() for h in controlled_m.group(1).split(',') if h.strip()]
        ps['controlled'].update(controlled_raw)
    if services_m:
        services_blob = services_m.group(1)
        try:
            ip_part, svc_part = services_blob.split(':', 1)
            ip = ip_part.strip()
            names = set(SERVICE_NAME_RE.findall(svc_part))
            if ip:
                ps['services'].setdefault(ip, set()).update(names)
        except ValueError:
            pass
    if data_m:
        # Placeholder for future data parsing
        pass


def assign_episodes(steps: List[Step]):
    """Assign episode numbers and per-episode step counters.

    Heuristic: an episode ends when a step has end=True in env observation JSON
    or when we detect a textual 'end=True' in the raw lines. The next concept
    action after an ended step starts a new episode. First episode is 1, and
    step_in_episode starts at 0 each episode as requested.
    """
    episode = 1
    step_counter = -1  # so first increment -> 0
    prev_ended = False
    for st in steps:
        if prev_ended:
            episode += 1
            step_counter = -1
            prev_ended = False
        step_counter += 1
        st.episode = episode
        st.step_in_episode = step_counter
        # Consider end flag OR a clearly positive reward (win) as episode boundary
        if st.end is True or (st.reward is not None and st.reward > 0):
            prev_ended = True


def extract_state(received: dict) -> dict:
    try:
        obs = received.get('observation') or {}
        state = obs.get('observation', {}).get('state')  # some logs? fallback next line
        if state is None:
            state = obs.get('state', {})
    except AttributeError:
        return {}
    return state or {}


def normalize_state(state: dict) -> dict:
    # Convert lists of objects to sets of printable tokens
    nets = {f"{n.get('ip')}/{n.get('mask')}" for n in state.get('known_networks', []) if isinstance(n, dict)}
    hosts = {h.get('ip') for h in state.get('known_hosts', []) if isinstance(h, dict)}
    controlled = {h.get('ip') for h in state.get('controlled_hosts', []) if isinstance(h, dict)}
    services_raw = state.get('known_services', {}) or {}
    services = {ip: {svc.get('name') for svc in svcs if isinstance(svc, dict)} for ip, svcs in services_raw.items()}
    data_raw = state.get('known_data', {}) or {}
    data_ids = {ip: {d.get('id') for d in ds if isinstance(d, dict)} for ip, ds in data_raw.items()}
    return dict(networks=nets, hosts=hosts, controlled=controlled, services=services, data=data_ids)


def compute_diffs(steps: List[Step]):
    prev_state: Dict[str, Any] = dict(
        networks=set(),
        hosts=set(),
        controlled=set(),
        services={},  # ip -> set(service names)
        data={},      # ip -> set(data ids)
    )
    for st in steps:
        # Prefer structured JSON state; fallback to parsed textual state
        if st.received:
            state = normalize_state(extract_state(st.received))
        elif st.parsed_state is not None:
            # Ensure copies so we don't mutate the original sets later inadvertently
            state = dict(
                networks=set(st.parsed_state.get('networks', set())),
                hosts=set(st.parsed_state.get('hosts', set())),
                controlled=set(st.parsed_state.get('controlled', set())),
                services={ip: set(svcs) for ip, svcs in (st.parsed_state.get('services') or {}).items()},
                data={ip: set(vals) for ip, vals in (st.parsed_state.get('data') or {}).items()},
            )
        else:
            continue
            # Hosts
            st.new_hosts = sorted(list(state['hosts'] - prev_state['hosts']))
            # Networks
            st.new_networks = sorted(list(state['networks'] - prev_state['networks']))
            # Services
            new_services: Dict[str, List[str]] = {}
            for ip, svcs in state['services'].items():
                prev_svcs = prev_state['services'].get(ip, set()) if isinstance(prev_state['services'], dict) else set()
                added = sorted(list(svcs - prev_svcs))
                if added:
                    new_services[ip] = added
            st.new_services = new_services
            # Data
            new_data: Dict[str, List[str]] = {}
            for ip, ds in state['data'].items():
                prev_ds = prev_state['data'].get(ip, set()) if isinstance(prev_state['data'], dict) else set()
                added = sorted(list(ds - prev_ds))
                if added:
                    new_data[ip] = added
            st.new_data = new_data
            prev_state = state


def style_reward(reward: Optional[int]) -> str:
    if reward is None:
        return 'white'
    if reward > 0:
        return 'bold green'
    if reward == 0:
        return 'bold yellow'
    return 'bold red'


def render_step(step: Step, show_json: bool = False) -> Panel:
    body_renderables: List[Any] = []

    # Header line under panel (concept vs real)
    concept_txt = Text(step.concept or '', style='bold cyan')
    real_txt = Text(step.real or '', style='bold magenta')
    header_line = Text()
    header_line.append('Concept: '); header_line.append(concept_txt)
    if step.real:
        header_line.append('\nReal   : '); header_line.append(real_txt)
    body_renderables.append(header_line)

    # Parameters table (from sending JSON)
    def _fmt_param(val: Any) -> Text:
        """Return a richly formatted value for parameter tables."""
        from rich.text import Text as _T
        t = _T()
        if isinstance(val, dict):
            # Network
            if 'ip' in val and 'mask' in val:
                t.append(f"{val.get('ip')}/{val.get('mask')}", style="yellow")
                return t
            # Host
            if 'ip' in val and 'mask' not in val:
                t.append(str(val.get('ip')), style="cyan")
                return t
            # Service object maybe
            if {'name','type','version'} <= set(val.keys()):
                t.append(val.get('name','?'), style='magenta')
                ver = val.get('version')
                if ver:
                    t.append(f" v{ver}", style='dim')
                if val.get('is_local'):
                    t.append(" (local)", style='green')
                return t
            # Fallback dict -> key: value lines
            inner = []
            for k,v in val.items():
                inner.append(f"{k}={v}")
            t.append(', '.join(inner), style='white')
            return t
        if isinstance(val, (list, tuple, set)):
            first = True
            for item in val:
                if not first:
                    t.append('\n')
                t.append_text(_fmt_param(item))
                first = False
            return t
        if isinstance(val, (str, int, float)):
            t.append(str(val), style='white')
            return t
        # Fallback generic JSON
        try:
            t.append(json.dumps(val, ensure_ascii=False), style='white')
        except Exception:
            t.append(repr(val), style='white')
        return t

    if step.sending:
        params = step.sending.get('parameters', {})
        if params:
            tbl = Table(title='Parameters', expand=False, box=None, show_header=True, header_style='bold blue', pad_edge=False)
            tbl.add_column('Key', style='cyan', no_wrap=True)
            tbl.add_column('Value', style='white')
            for k, v in params.items():
                tbl.add_row(str(k), _fmt_param(v))
            body_renderables.append(tbl)

    # Env response summary
    # Build env state view (JSON structured or textual parsed)
    if step.received or step.parsed_state is not None:
        if step.received:
            received = step.received
            obs_state = normalize_state(extract_state(received))
        else:
            # Already similar format; ensure parsed_state exists
            ps = step.parsed_state or {}
            obs_state = dict(
                networks=ps.get('networks', set()),
                hosts=ps.get('hosts', set()),
                controlled=ps.get('controlled', set()),
                services=ps.get('services', {}),
                data=ps.get('data', {}),
            )
        # Detailed multiline listing instead of counts
        detail_text = Text()
        networks = sorted(obs_state['networks'])
        hosts = sorted(obs_state['hosts'])
        controlled = set(obs_state['controlled'])
        services = obs_state['services']
        data_items = obs_state['data']

        # Networks
        detail_text.append("Networks (" + str(len(networks)) + ")\n", style="bold yellow")
        for n in networks:
            detail_text.append(f"  • {n}\n", style="yellow")

        # Hosts (mark controlled)
        detail_text.append("Hosts (" + str(len(hosts)) + ")\n", style="bold cyan")
        for h in hosts:
            mark = " *" if h in controlled else ""
            style = "bold green" if h in controlled else "cyan"
            detail_text.append(f"  • {h}{mark}\n", style=style)

        # Controlled (explicit list)
        detail_text.append("Controlled Hosts (" + str(len(controlled)) + ")\n", style="bold green")
        for ch in sorted(controlled):
            detail_text.append(f"  • {ch}\n", style="green")

        # Services per host
        total_services = sum(len(v) for v in services.values())
        detail_text.append(f"Services ({total_services})\n", style="bold magenta")
        for ip in sorted(services.keys()):
            svc_list = sorted(list(services[ip]))
            detail_text.append(f"  • {ip}\n", style="magenta")
            for svc in svc_list:
                detail_text.append(f"      - {svc}\n", style="bright_magenta")

        # Data items per host
        total_data = sum(len(v) for v in data_items.values())
        detail_text.append(f"Data Items ({total_data})\n", style="bold blue")
        for ip in sorted(data_items.keys()):
            ids = sorted(list(data_items[ip]))
            detail_text.append(f"  • {ip}\n", style="blue")
            for di in ids:
                detail_text.append(f"      - {di}\n", style="bright_blue")

        body_renderables.append(Panel(detail_text, title="Environment State", border_style="yellow"))

    # Diffs section
    diffs_lines = []
    if step.new_hosts:
        diffs_lines.append(f"[green]+Hosts[/]: {', '.join(step.new_hosts)}")
    if step.new_networks:
        diffs_lines.append(f"[green]+Networks[/]: {', '.join(step.new_networks)}")
    if step.new_services:
        svc_parts = []
        for ip, names in step.new_services.items():
            svc_parts.append(f"{ip}: {'; '.join(names)}")
        diffs_lines.append(f"[green]+Services[/]: {' | '.join(svc_parts)}")
    if step.new_data:
        data_parts = []
        for ip, ids in step.new_data.items():
            data_parts.append(f"{ip}: {'; '.join(ids)}")
        diffs_lines.append(f"[green]+Data[/]: {' | '.join(data_parts)}")
    if diffs_lines:
        from rich.text import Text as _T
        body_renderables.append(_T.from_markup('\n'.join(diffs_lines)))

    if show_json and step.sending:
        body_renderables.append(Panel(Syntax(json.dumps(step.sending, indent=2), 'json', theme='monokai', word_wrap=False), title='Sending JSON', border_style='cyan'))
    if show_json and step.received:
        body_renderables.append(Panel(Syntax(json.dumps(step.received, indent=2)[:4000], 'json', theme='monokai', word_wrap=False), title='Received JSON (truncated)', border_style='magenta'))

    reward_str = f"reward {step.reward}" if step.reward is not None else 'reward ?'
    title = f"Ep {step.episode} Step {step.step_in_episode} (#{step.index}) | {step.timestamp} | {reward_str}"
    return Panel(Group(*body_renderables), title=title, border_style=style_reward(step.reward))


def render_summary(steps: List[Step], console: Console, args):
    shown = 0
    for st in steps:
        if args.search and args.search.lower() not in (st.concept or '').lower() and (
            not st.real or args.search.lower() not in st.real.lower()
        ):
            continue
        console.print(render_step(st, show_json=args.show_json))
        shown += 1
        if args.limit and shown >= args.limit:
            break
    if shown == 0:
        console.print('[bold yellow]No steps matched filters[/]')


def render_raw(path: Path, console: Console):
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        console.print(line)


def build_arg_parser():
    p = argparse.ArgumentParser(description='Rich summary / raw viewer for q_agent.log')
    p.add_argument('log_file')
    p.add_argument('--mode', choices=['summary', 'raw'], default='summary', help='View mode')
    p.add_argument('--limit', type=int, help='Limit number of steps (summary mode)')
    p.add_argument('--search', help='Substring filter (summary mode)')
    p.add_argument('--show-json', action='store_true', help='Include raw JSON inside panels')
    p.add_argument('--force-color', action='store_true', help='Force color when not a TTY')
    p.add_argument('--no-color', action='store_true', help='Disable color entirely')
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    path = Path(args.log_file).expanduser().resolve()
    if not path.exists():
        Console().print(f"[bold red]File not found:[/] {path}")
        return 2

    if args.no_color:
        console = Console(no_color=True)
    else:
        auto_force = not sys.stdout.isatty()
        force = args.force_color or auto_force
        console = Console(force_terminal=force, color_system='truecolor' if force else None)

    if args.mode == 'raw':
        render_raw(path, console)
        return 0

    steps = parse_log_lines(path)
    assign_episodes(steps)
    compute_diffs(steps)
    render_summary(steps, console, args)
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
