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
from typing import Dict, List, Optional, Iterable, Any, Set

from rich.console import Console, Group
from rich.columns import Columns
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
INITIAL_STATE_RE = re.compile(r"\[\+\]\s+Initial state before first action:(.*)")
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
REAL_STATE_SERVICES_RE = re.compile(r"I2C: Real state known services: \{(.*)\}$")
REAL_STATE_DATA_RE = re.compile(r"I2C: Real state known data: \{(.*)\}$")
CONCEPT_STATE_SERVICES_RE = re.compile(r"I2C: New concept known_services: \{(.*)\}$")
CONCEPT_STATE_DATA_RE = re.compile(r"I2C: New concept known_data: \{(.*)\}$")
CONCEPT_STATE_KNOWN_HOSTS_RE = re.compile(r"I2C: New concept known_hosts: \{(.*)\}$")
CONCEPT_STATE_CONTROLLED_HOSTS_RE = re.compile(r"I2C: New concept controlled_hosts: \{(.*)\}$")
CONCEPT_STATE_NETWORKS_RE = re.compile(r"I2C: New concept known_nets: \{(.*)\}$")

# Service name extraction inside service listing
SERVICE_NAME_RE = re.compile(r"Service\(name='([^']+)'")
SERVICES_HOST_RE = re.compile(r"([^:]+):\s*\{([^}]*)\}")
DATA_HOST_RE = re.compile(r"([^:]+):\s*\{([^}]*)\}")
DATA_ENTRY_RE = re.compile(r"Data\(([^)]*)\)")
DATA_FIELD_RE = re.compile(r"(owner|id|size|type)=('([^']*)'|[^,]+)")
CONCEPT_KV_RE = re.compile(r"'([^']+)'\s*:\s*([^,}]+)")

def _clean_host_token(token: str) -> str:
    """Normalize a host key parsed from textual dictionaries.

    Removes leading commas/spaces and surrounding quotes.
    """
    s = token.strip()
    if s.startswith(','):
        s = s[1:].strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1].strip()
    return s


def _format_data_summary(data_id: Optional[Any], owner: Optional[Any], dtype: Optional[Any], size: Optional[Any]) -> str:
    """Create a concise printable summary for a data item."""
    label = str(data_id) if data_id not in (None, '') else '?'
    extras: List[str] = []
    if owner not in (None, ''):
        extras.append(f"owner={owner}")
    if dtype not in (None, ''):
        extras.append(f"type={dtype}")
    if size not in (None, ''):
        extras.append(f"size={size}")
    return f"{label} ({', '.join(extras)})" if extras else label


def _parse_textual_known_data(blob: str) -> Dict[str, Set[str]]:
    """Parse textual representation of known data into host -> set(data summaries)."""
    results: Dict[str, Set[str]] = {}
    for match in DATA_HOST_RE.finditer(blob):
        host = _clean_host_token(match.group(1))
        payload = match.group(2).strip()
        entries: Set[str] = set()
        if payload:
            for data_match in DATA_ENTRY_RE.finditer(payload):
                fields_str = data_match.group(1)
                details: Dict[str, Any] = {}
                for name, raw_value, quoted_value in DATA_FIELD_RE.findall(fields_str):
                    value = quoted_value if quoted_value else raw_value.strip()
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    details[name] = value
                summary = _format_data_summary(
                    details.get('id'),
                    details.get('owner'),
                    details.get('type'),
                    details.get('size'),
                )
                entries.add(summary)
        results[host] = entries
    return results


def _format_data_object(obj: Any) -> str:
    """Normalize JSON data objects coming from the observation."""
    if isinstance(obj, dict):
        return _format_data_summary(
            obj.get('id') or obj.get('data_id'),
            obj.get('owner'),
            obj.get('type'),
            obj.get('size'),
        )
    return str(obj)


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

            # If we haven't started a step yet, look for the initial state
            # information (either explicit "Initial state..." log or the first
            # I2C real-state dump after registration) and create a synthetic
            # step for it so it is visible in the summary.
            if current is None:
                initial_m = INITIAL_STATE_RE.search(msg)
                has_real_state = any(
                    pat.search(msg)
                    for pat in (
                        REAL_STATE_NETS_RE,
                        REAL_STATE_HOSTS_RE,
                        REAL_STATE_CONTROLLED_RE,
                        REAL_STATE_SERVICES_RE,
                        REAL_STATE_DATA_RE,
                    )
                )
                if initial_m or has_real_state:
                    label = "Initial state before first action"
                    current = Step(index=len(steps)+1, timestamp=last_ts, concept=label)
                    steps.append(current)
                    current.raw_lines.append(raw_line)
                    _maybe_parse_textual_state(current, msg)
                    continue
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
            # Handle initial state logs and I2C state lines that appear before the
            # first Concept Action. We treat them as a dedicated "initial state"
            # step so the first environment state (before any action) is visible.
            if current is None:
                initial_m = INITIAL_STATE_RE.search(raw_line)
                has_real_state = any(
                    pat.search(raw_line)
                    for pat in (
                        REAL_STATE_NETS_RE,
                        REAL_STATE_HOSTS_RE,
                        REAL_STATE_CONTROLLED_RE,
                        REAL_STATE_SERVICES_RE,
                        REAL_STATE_DATA_RE,
                    )
                )
                if initial_m or has_real_state:
                    # Create a synthetic step 0 representing the initial state.
                    label = "Initial state before first action"
                    current = Step(index=len(steps)+1, timestamp=last_ts, concept=label)
                    steps.append(current)
                    current.raw_lines.append(raw_line)
                    _maybe_parse_textual_state(current, raw_line)
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
    nets_m = REAL_STATE_NETS_RE.search(line)
    hosts_m = REAL_STATE_HOSTS_RE.search(line)
    controlled_m = REAL_STATE_CONTROLLED_RE.search(line)
    services_m = REAL_STATE_SERVICES_RE.search(line)
    data_m = REAL_STATE_DATA_RE.search(line)
    concept_services_m = CONCEPT_STATE_SERVICES_RE.search(line)
    concept_data_m = CONCEPT_STATE_DATA_RE.search(line)
    concept_hosts_m = CONCEPT_STATE_KNOWN_HOSTS_RE.search(line) or CONCEPT_STATE_CONTROLLED_HOSTS_RE.search(line)
    concept_networks_m = CONCEPT_STATE_NETWORKS_RE.search(line)
    concept_controlled_m = CONCEPT_STATE_CONTROLLED_HOSTS_RE.search(line)
    if not any([
        nets_m,
        hosts_m,
        controlled_m,
        services_m,
        data_m,
        concept_services_m,
        concept_data_m,
        concept_hosts_m,
        concept_networks_m,
        concept_controlled_m,
    ]):
        return
    # Now initialize container since something matched
    if step.parsed_state is None:
        step.parsed_state = {
            # Real (IP-level) view reconstructed from INFO logs
            'networks': set(),
            'hosts': set(),
            'controlled': set(),
            'services': {},   # ip -> set(service_name)
            'data': {},       # ip -> set(data_summaries)
            # Concept-level view parsed from I2C: New concept ... lines
            'concept_hosts_map': {},   # concept_host -> real_ip (when known)
            'concept_services': {},    # concept_host -> set(service_name)
            'concept_data': {},        # concept_host -> set(data_summaries)
            'concept_networks': {},    # concept_net -> real_network_str
            'concept_controlled_hosts': set(),  # set of concept host keys
        }
    ps = step.parsed_state
    # Concept known_hosts / controlled_hosts mapping (concept -> IP)
    if concept_hosts_m:
        blob = concept_hosts_m.group(1)
        mapping: Dict[str, str] = {}
        for k, v in CONCEPT_KV_RE.findall(blob):
            key = _clean_host_token(k)
            val = _clean_host_token(v)
            mapping[key] = val
        ps['concept_hosts_map'].update(mapping)
    # Concept controlled hosts (track which concepts are controlled)
    if concept_controlled_m:
        blob = concept_controlled_m.group(1)
        for k, _ in CONCEPT_KV_RE.findall(blob):
            key = _clean_host_token(k)
            ps['concept_controlled_hosts'].add(key)
    if nets_m:
        nets_raw = [n.strip() for n in nets_m.group(1).split(',') if n.strip()]
        ps['networks'].update(nets_raw)
    if hosts_m:
        hosts_raw = [h.strip() for h in hosts_m.group(1).split(',') if h.strip()]
        ps['hosts'].update(hosts_raw)
    if controlled_m:
        controlled_raw = [h.strip() for h in controlled_m.group(1).split(',') if h.strip()]
        ps['controlled'].update(controlled_raw)
    # Real-state services
    if services_m:
        services_blob = services_m.group(1)
        try:
            for host_match in SERVICES_HOST_RE.finditer(services_blob):
                ip = _clean_host_token(host_match.group(1))
                svc_part = host_match.group(2)
                names = set(SERVICE_NAME_RE.findall(svc_part))
                if ip:
                    ps['services'].setdefault(ip, set()).update(names)
        except Exception:
            pass
    # Concept-level services (fallback when real-state is absent)
    if concept_services_m:
        services_blob = concept_services_m.group(1)
        try:
            for host_match in SERVICES_HOST_RE.finditer(services_blob):
                host_key = _clean_host_token(host_match.group(1))
                svc_part = host_match.group(2)
                names = set(SERVICE_NAME_RE.findall(svc_part))
                if host_key:
                    ps['concept_services'].setdefault(host_key, set()).update(names)
        except Exception:
            pass
    # Real-state data
    if data_m:
        parsed = _parse_textual_known_data(data_m.group(1))
        for host, items in parsed.items():
            bucket = ps['data'].setdefault(host, set())
            bucket.update(items)
    # Concept-level data
    if concept_data_m:
        parsed = _parse_textual_known_data(concept_data_m.group(1))
        for host_key, items in parsed.items():
            bucket = ps['concept_data'].setdefault(host_key, set())
            bucket.update(items)
    # Concept-level networks
    if concept_networks_m:
        blob = concept_networks_m.group(1)
        # Example blob: net_0_0hosts/24: 192.168.93.0/24, net_1_1hosts/24: 192.168.94.0/24
        # Keys may not be quoted, so we cannot rely on CONCEPT_KV_RE here.
        for pair in blob.split(','):
            if ':' not in pair:
                continue
            key_str, val_str = pair.split(':', 1)
            key = _clean_host_token(key_str)
            val = _clean_host_token(val_str)
            if key:
                ps['concept_networks'][key] = val


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
    nets = {f"{n.get('ip')}/{n.get('mask')}" for n in state.get('known_networks', []) if isinstance(n, dict) and n.get('ip') and n.get('mask')}
    hosts = {h.get('ip') for h in state.get('known_hosts', []) if isinstance(h, dict) and h.get('ip')}
    controlled = {h.get('ip') for h in state.get('controlled_hosts', []) if isinstance(h, dict) and h.get('ip')}
    services_raw = state.get('known_services', {}) or {}
    services = {ip: {svc.get('name') for svc in svcs if isinstance(svc, dict)} for ip, svcs in services_raw.items()}
    data_raw = state.get('known_data', {}) or {}
    data_items: Dict[str, Set[str]] = {}

    def _iterate_data(ds: Any) -> Iterable:
        if isinstance(ds, dict):
            return ds.values()
        if isinstance(ds, (list, tuple, set)):
            return ds
        if ds is None:
            return []
        return [ds]

    for ip, ds in data_raw.items():
        key = str(ip)
        entries: Set[str] = set()
        for item in _iterate_data(ds):
            entries.add(_format_data_object(item))
        data_items[key] = entries
    return dict(networks=nets, hosts=hosts, controlled=controlled, services=services, data=data_items)


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
    # Build env state view by merging JSON (if present) with parsed textual fragments
    if step.received or step.parsed_state is not None:
        obs_state = dict(networks=set(), hosts=set(), controlled=set(), services={}, data={})
        if step.received:
            js = normalize_state(extract_state(step.received))
            obs_state['networks'] |= set(js.get('networks', set()))
            obs_state['hosts'] |= set(js.get('hosts', set()))
            obs_state['controlled'] |= set(js.get('controlled', set()))
            for ip, svcs in (js.get('services') or {}).items():
                obs_state['services'].setdefault(ip, set()).update(svcs)
            for ip, items in (js.get('data') or {}).items():
                obs_state['data'].setdefault(ip, set()).update(items)
        if step.parsed_state is not None:
            ps = step.parsed_state
            obs_state['networks'] |= set(ps.get('networks', set()))
            obs_state['hosts'] |= set(ps.get('hosts', set()))
            obs_state['controlled'] |= set(ps.get('controlled', set()))
            for ip, svcs in (ps.get('services') or {}).items():
                obs_state['services'].setdefault(ip, set()).update(svcs)
            for ip, items in (ps.get('data') or {}).items():
                obs_state['data'].setdefault(ip, set()).update(items)

        # ---------------- Real (IP-level) environment view ----------------
        detail_text = Text()
        networks = sorted(obs_state['networks'])
        hosts = sorted(obs_state['hosts'])
        controlled = set(obs_state['controlled'])
        services = obs_state['services']
        data_items = obs_state['data']

        # Networks
        detail_text.append(f"Networks ({len(networks)})\n", style="bold yellow")
        for n in networks:
            detail_text.append(f"  • {n}\n", style="yellow")

        # Hosts (mark controlled)
        detail_text.append(f"Hosts ({len(hosts)})\n", style="bold cyan")
        for h in hosts:
            mark = " *" if h in controlled else ""
            style = "bold green" if h in controlled else "cyan"
            detail_text.append(f"  • {h}{mark}\n", style=style)

        # Controlled (explicit list)
        detail_text.append(f"Controlled Hosts ({len(controlled)})\n", style="bold green")
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

        real_panel = Panel(detail_text, title="Environment State (Real)", border_style="yellow")

        # ---------------- Concept-level view (kept separate) ----------------
        concept_panel = None
        if step.parsed_state is not None:
            ps = step.parsed_state
            concept_hosts_map = ps.get('concept_hosts_map') or {}
            concept_services = ps.get('concept_services') or {}
            concept_data = ps.get('concept_data') or {}
            concept_networks = ps.get('concept_networks') or {}
            concept_controlled = ps.get('concept_controlled_hosts') or set()

            if concept_hosts_map or concept_services or concept_data or concept_networks:
                concept_text = Text()

                if concept_networks:
                    concept_text.append(f"Concept Networks ({len(concept_networks)})\n", style="bold yellow")
                    for cn, real_net in sorted(concept_networks.items(), key=lambda kv: kv[0]):
                        if real_net:
                            concept_text.append(f"  • {cn} → {real_net}\n", style="yellow")
                        else:
                            concept_text.append(f"  • {cn}\n", style="yellow")

                if concept_hosts_map:
                    concept_text.append("Concept Hosts\n", style="bold cyan")
                    for ch, ip in sorted(concept_hosts_map.items(), key=lambda kv: kv[0]):
                        mark = " *" if ch in concept_controlled else ""
                        if ip:
                            concept_text.append(f"  • {ch}{mark} → {ip}\n", style="cyan")
                        else:
                            concept_text.append(f"  • {ch}{mark}\n", style="cyan")

                if concept_controlled:
                    concept_text.append(f"Concept Controlled Hosts ({len(concept_controlled)})\n", style="bold green")
                    for ch in sorted(concept_controlled):
                        target_ip = concept_hosts_map.get(ch)
                        if target_ip:
                            concept_text.append(f"  • {ch} → {target_ip}\n", style="green")
                        else:
                            concept_text.append(f"  • {ch}\n", style="green")

                if concept_services:
                    total_c_services = sum(len(v) for v in concept_services.values())
                    concept_text.append(f"Concept Services ({total_c_services})\n", style="bold magenta")
                    for ch in sorted(concept_services.keys()):
                        svc_list = sorted(list(concept_services[ch]))
                        concept_text.append(f"  • {ch}\n", style="magenta")
                        for svc in svc_list:
                            concept_text.append(f"      - {svc}\n", style="bright_magenta")

                if concept_data:
                    total_c_data = sum(len(v) for v in concept_data.values())
                    concept_text.append(f"Concept Data ({total_c_data})\n", style="bold blue")
                    for ch in sorted(concept_data.keys()):
                        ids = sorted(list(concept_data[ch]))
                        concept_text.append(f"  • {ch}\n", style="blue")
                        for di in ids:
                            concept_text.append(f"      - {di}\n", style="bright_blue")

                concept_panel = Panel(concept_text, title="Concept State", border_style="cyan")

        # Layout: show real and concept state side by side when both exist
        if concept_panel is not None:
            body_renderables.append(Columns([real_panel, concept_panel], equal=True, expand=True))
        else:
            body_renderables.append(real_panel)

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
