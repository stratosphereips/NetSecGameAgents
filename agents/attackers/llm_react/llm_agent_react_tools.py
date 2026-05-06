"""
Raw ReAct baseline agent for NetSecGame.

Comparison baseline for the SGRL (planner + symbolic monitor + tactical
executor) architecture. This agent exposes NetSecGame's 5 primitive actions
as tool functions and lets a single LLM drive the full episode end-to-end:
no planner, no monitor, no segment boundaries.

Designed for the Phase 1 reliability comparison described in
docs/ideas/2026-05-05-verifier-experiments.md.
"""

import argparse
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import netaddr

from netsecgame import (
    Action,
    Observation,
    AgentRole,
    BaseAgent,
    generate_valid_actions,
)
from netsecgame.game_components import (
    ActionType,
    AgentStatus,
    GameState,
)
from utils import RunStats, filter_log_files_from_state

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# OpenAI-format tool definitions for the 5 NetSecGame primitives.
NETSECGAME_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "scan_network",
            "description": (
                "Scan a network to discover live hosts. Must be called from a "
                "controlled host."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_host": {
                        "type": "string",
                        "description": "IP of the controlled host to scan from.",
                    },
                    "target_network": {
                        "type": "string",
                        "description": "CIDR of the network to scan (e.g. 192.168.1.0/24).",
                    },
                },
                "required": ["source_host", "target_network"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_services",
            "description": (
                "Enumerate services running on a known host. Must be called "
                "from a controlled host."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_host": {"type": "string"},
                    "target_host": {"type": "string"},
                },
                "required": ["source_host", "target_host"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exploit_service",
            "description": (
                "Exploit a service on a known host to gain control of that "
                "host. Service must be known on the target."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_host": {"type": "string"},
                    "target_host": {"type": "string"},
                    "service_name": {
                        "type": "string",
                        "description": "Name of the service to exploit (e.g. 'openssh').",
                    },
                },
                "required": ["source_host", "target_host", "service_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_data",
            "description": (
                "Search a controlled host for data items. Target must be in "
                "controlled_hosts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_host": {"type": "string"},
                    "target_host": {"type": "string"},
                },
                "required": ["source_host", "target_host"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exfiltrate_data",
            "description": (
                "Exfiltrate a known data item from one controlled host to "
                "another. Mission success typically requires exfiltrating to a "
                "public/external IP."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_host": {
                        "type": "string",
                        "description": "Controlled host where the data is located.",
                    },
                    "target_host": {
                        "type": "string",
                        "description": "Controlled host (often public/external) to receive the data.",
                    },
                    "data_owner": {
                        "type": "string",
                        "description": "Owner field of the Data tuple (as shown in state).",
                    },
                    "data_id": {
                        "type": "string",
                        "description": "Identifier field of the Data tuple (as shown in state).",
                    },
                },
                "required": ["source_host", "target_host", "data_owner", "data_id"],
            },
        },
    },
]


SYSTEM_PROMPT = """You are a cyber security red team agent operating in the NetSecGame simulation.

Your overall mission: {mission}

Mechanics:
- You can only act from hosts you control.
- To discover hosts in a network you must scan_network.
- To exploit_service you must first find_services on the target.
- To find_data on a host you must first control it.
- exfiltrate_data is the typical mission completion: move data to an external/public IP that you also control.

Each turn you will be shown the current state and the currently valid actions. Reason briefly about the best next action, then call exactly one tool.
"""


def _is_public(ip) -> bool:
    try:
        return not netaddr.IPAddress(str(ip)).is_private()
    except Exception:
        return False


def serialize_state(state: GameState) -> str:
    """Render the GameState as a compact text summary for the LLM."""
    lines = ["CURRENT STATE:"]

    nets = sorted(state.known_networks, key=lambda x: str(x))
    lines.append(
        "  Networks known: " + (", ".join(str(n) for n in nets) if nets else "(none)")
    )

    all_hosts = sorted(
        set(state.known_hosts).union(state.controlled_hosts), key=lambda x: str(x)
    )
    host_descs = []
    for h in all_hosts:
        flags = []
        if h in state.controlled_hosts:
            flags.append("controlled")
        flags.append("public" if _is_public(h) else "private")
        host_descs.append(f"{h} [{', '.join(flags)}]")
    lines.append(
        "  Hosts known: " + (", ".join(host_descs) if host_descs else "(none)")
    )

    svc_lines = []
    for h in sorted(state.known_services.keys(), key=lambda x: str(x)):
        for s in sorted(state.known_services[h], key=lambda x: str(x)):
            svc_lines.append(f"{h}: {s.name}")
    lines.append(
        "  Services known: " + (", ".join(svc_lines) if svc_lines else "(none)")
    )

    data_lines = []
    for h in sorted(state.known_data.keys(), key=lambda x: str(x)):
        for d in sorted(state.known_data[h], key=lambda x: str(x)):
            data_lines.append(f"{h}: ({d.owner}, {d.id})")
    lines.append("  Data known: " + (", ".join(data_lines) if data_lines else "(none)"))

    return "\n".join(lines)


def render_valid_actions(state: GameState, limit: int = 30) -> str:
    """Render currently-valid actions as a full enumerated hint list."""
    valid = generate_valid_actions(state, include_blocks=False)
    if not valid:
        return "  (no valid actions)"
    lines = []
    for a in valid[:limit]:
        params = ", ".join(f"{k}={v}" for k, v in (a.parameters or {}).items())
        lines.append(f"  - {a.type.name}({params})")
    if len(valid) > limit:
        lines.append(f"  ... and {len(valid) - limit} more")
    return "\n".join(lines)


def render_valid_actions_summary(state: GameState) -> str:
    """Render only the count of valid actions per type — O(1) in size."""
    from collections import Counter

    valid = generate_valid_actions(state, include_blocks=False)
    if not valid:
        return "  (no valid actions)"
    counts = Counter(a.type.name for a in valid)
    parts = [f"{n} {name}" for name, n in sorted(counts.items())]
    return f"  Total: {len(valid)} valid actions ({', '.join(parts)})"


@dataclass
class EpisodeOutcome:
    won: bool
    detected: bool
    steps: int
    input_tokens: int
    output_tokens: int
    end_reason: str
    total_reward: float = 0.0
    transcript: list = field(default_factory=list)


class ReActAgent(BaseAgent):
    """LLM driving NetSecGame primitives directly via tool use."""

    def __init__(
        self,
        host: str,
        port: int,
        role: AgentRole,
        model: str,
        api_key: str = "EMPTY",
        base_url: Optional[str] = None,
        mission: Optional[str] = None,
        max_steps: int = 100,
        max_input_tokens: int = 200_000,
        max_output_tokens_per_call: int = 500,
        temperature: float = 0.2,
        show_valid_actions: str = "none",
        valid_actions_limit: int = 30,
        action_memory_size: int = 0,
    ):
        super().__init__(host, port, role)
        if OpenAI is None:
            raise ImportError("openai package required for ReActAgent")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.mission = mission or (
            "Compromise hosts in the network and exfiltrate any sensitive data "
            "to an external (public) host you control."
        )
        self.max_steps = max_steps
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens_per_call = max_output_tokens_per_call
        self.temperature = temperature
        if show_valid_actions not in ("none", "summary", "full"):
            raise ValueError(
                f"show_valid_actions must be one of none/summary/full, got {show_valid_actions!r}"
            )
        self.show_valid_actions = show_valid_actions
        self.valid_actions_limit = valid_actions_limit
        self.action_memory_size = max(0, int(action_memory_size))

        # Per-episode mutable state
        self._messages: list = []
        self._input_tokens = 0
        self._output_tokens = 0
        self._action_memory: deque = deque(maxlen=self.action_memory_size or None)
        self._prev_state_sig: Optional[tuple] = None

    # ------------------------------------------------------------------
    # Tool-call parsing
    # ------------------------------------------------------------------

    def _find_host(self, state: GameState, ip_str: str):
        for h in set(state.known_hosts).union(state.controlled_hosts):
            if str(h) == ip_str:
                return h
        return None

    def _find_network(self, state: GameState, cidr_str: str):
        for n in state.known_networks:
            if str(n) == cidr_str:
                return n
        return None

    def _find_service(self, state: GameState, host_ip, service_name: str):
        host = self._find_host(state, str(host_ip))
        if host is None:
            return None
        services = list(state.known_services.get(host, set()))
        # Exact match first.
        for s in services:
            if s.name == service_name:
                return s
        # Lenient: case-insensitive substring match. Only succeed if unique.
        needle = service_name.lower().strip()
        matches = [s for s in services if needle in s.name.lower()]
        if len(matches) == 1:
            return matches[0]
        return None

    def _find_data(self, state: GameState, host_ip, owner: str, ident: str):
        host = self._find_host(state, str(host_ip))
        if host is None:
            return None
        for d in state.known_data.get(host, set()):
            if d.owner == owner and d.id == ident:
                return d
        return None

    def parse_tool_call(
        self, name: str, args: dict, state: GameState
    ) -> tuple[Optional[Action], Optional[str]]:
        """Map a tool call into an Action. Returns (action, error_msg).

        On success, error_msg is None. On failure, action is None and error_msg
        explains what was wrong with enough specificity for the LLM to correct.
        """
        try:
            if name == "scan_network":
                src = self._find_host(state, args["source_host"])
                net = self._find_network(state, args["target_network"])
                if src is None:
                    return None, (
                        f"source_host {args['source_host']!r} not in known/controlled hosts"
                    )
                if net is None:
                    known = sorted(str(n) for n in state.known_networks)
                    return None, (
                        f"target_network {args['target_network']!r} not in known networks: {known}"
                    )
                return (
                    Action(
                        ActionType.ScanNetwork,
                        parameters={"source_host": src, "target_network": net},
                    ),
                    None,
                )

            if name == "find_services":
                src = self._find_host(state, args["source_host"])
                tgt = self._find_host(state, args["target_host"])
                if src is None:
                    return (
                        None,
                        f"source_host {args['source_host']!r} not in known/controlled hosts",
                    )
                if tgt is None:
                    return (
                        None,
                        f"target_host {args['target_host']!r} not in known hosts",
                    )
                return (
                    Action(
                        ActionType.FindServices,
                        parameters={"source_host": src, "target_host": tgt},
                    ),
                    None,
                )

            if name == "exploit_service":
                src = self._find_host(state, args["source_host"])
                tgt = self._find_host(state, args["target_host"])
                if src is None:
                    return (
                        None,
                        f"source_host {args['source_host']!r} not in known/controlled hosts",
                    )
                if tgt is None:
                    return (
                        None,
                        f"target_host {args['target_host']!r} not in known hosts",
                    )
                svc = self._find_service(
                    state, args["target_host"], args["service_name"]
                )
                if svc is None:
                    known = sorted(s.name for s in state.known_services.get(tgt, set()))
                    if not known:
                        return None, (
                            f"no services known on {tgt}; run find_services on it first"
                        )
                    return None, (
                        f"service {args['service_name']!r} not found on {tgt}; "
                        f"known services there: {known}"
                    )
                return (
                    Action(
                        ActionType.ExploitService,
                        parameters={
                            "source_host": src,
                            "target_host": tgt,
                            "target_service": svc,
                        },
                    ),
                    None,
                )

            if name == "find_data":
                src = self._find_host(state, args["source_host"])
                tgt = self._find_host(state, args["target_host"])
                if src is None:
                    return (
                        None,
                        f"source_host {args['source_host']!r} not in known/controlled hosts",
                    )
                if tgt is None:
                    return (
                        None,
                        f"target_host {args['target_host']!r} not in known hosts",
                    )
                return (
                    Action(
                        ActionType.FindData,
                        parameters={"source_host": src, "target_host": tgt},
                    ),
                    None,
                )

            if name == "exfiltrate_data":
                src = self._find_host(state, args["source_host"])
                tgt = self._find_host(state, args["target_host"])
                if src is None:
                    return (
                        None,
                        f"source_host {args['source_host']!r} not in known/controlled hosts",
                    )
                if tgt is None:
                    return (
                        None,
                        f"target_host {args['target_host']!r} not in known hosts",
                    )
                data = self._find_data(
                    state, args["source_host"], args["data_owner"], args["data_id"]
                )
                if data is None:
                    known = sorted(
                        f"({d.owner}, {d.id})" for d in state.known_data.get(src, set())
                    )
                    if not known:
                        return None, (
                            f"no data items known on {src}; run find_data on it first"
                        )
                    return None, (
                        f"data ({args['data_owner']!r}, {args['data_id']!r}) "
                        f"not found on {src}; known data on that host: {known}"
                    )
                return (
                    Action(
                        ActionType.ExfiltrateData,
                        parameters={
                            "source_host": src,
                            "target_host": tgt,
                            "data": data,
                        },
                    ),
                    None,
                )
        except (KeyError, TypeError) as e:
            self.logger.warning(f"Tool call parse failure for {name}: {e}")
            return None, f"malformed arguments for {name}: {e}"
        return None, f"unknown tool name {name!r}"

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------

    def _reset_episode(self):
        self._messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(mission=self.mission)}
        ]
        self._input_tokens = 0
        self._output_tokens = 0
        self._action_memory = deque(maxlen=self.action_memory_size or None)
        self._prev_state_sig = None

    @staticmethod
    def _state_signature(state: GameState) -> tuple:
        """Compact signature capturing whether progress was made.

        Captures the size of each 'known' / 'controlled' set. A change after an
        action means the action produced new information or a new foothold —
        i.e. it was helpful. No change means the action was a no-op for
        progress purposes.
        """
        return (
            len(state.known_networks),
            len(state.known_hosts),
            len(state.controlled_hosts),
            sum(len(v) for v in state.known_services.values()),
            sum(len(v) for v in state.known_data.values()),
        )

    def _format_action_memory(self) -> str:
        if not self._action_memory:
            return ""
        lines = ["Recent actions (most recent last):"]
        for step, action_repr, evaluation in self._action_memory:
            lines.append(f"  step {step}: {action_repr} -> {evaluation}")
        lines.append(
            "Avoid repeating actions marked 'unhelpful' or 'invalid' with the same arguments."
        )
        return "\n".join(lines)

    def _build_user_message(
        self, state: GameState, step: int, last_result: Optional[str]
    ) -> str:
        parts = [f"[Step {step}]"]
        if last_result:
            parts.append(f"Result of previous action: {last_result}")
        parts.append(serialize_state(state))
        if self.show_valid_actions == "full":
            parts.append("Currently valid actions:")
            parts.append(render_valid_actions(state, limit=self.valid_actions_limit))
        elif self.show_valid_actions == "summary":
            parts.append("Valid action counts:")
            parts.append(render_valid_actions_summary(state))
        memory_block = self._format_action_memory()
        if memory_block:
            parts.append(memory_block)
        parts.append(
            "Reason briefly (1-3 sentences) about the next action, then call exactly one tool."
        )
        return "\n\n".join(parts)

    def _llm_step(self) -> tuple[Optional[str], Optional[dict], str]:
        """Single LLM call. Returns (tool_name, tool_args, reasoning_text)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages,
            tools=NETSECGAME_TOOLS,
            tool_choice="required",
            temperature=self.temperature,
            max_tokens=self.max_output_tokens_per_call,
        )
        if response.usage:
            self._input_tokens += response.usage.prompt_tokens or 0
            self._output_tokens += response.usage.completion_tokens or 0

        msg = response.choices[0].message
        # Append the assistant message so the next call has context
        self._messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (msg.tool_calls or [])
                ],
            }
        )
        if not msg.tool_calls:
            return None, None, msg.content or ""
        tc = msg.tool_calls[0]
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        return tc.function.name, args, msg.content or ""

    def _append_tool_result(self, last_tool_call_id: str, result: str):
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": last_tool_call_id,
                "content": result,
            }
        )

    def run_episode(
        self, observation: Observation, verbose: bool = False
    ) -> EpisodeOutcome:
        self._reset_episode()

        transcript = []
        last_result: Optional[str] = None
        end_reason = "unknown"
        step = 0
        total_reward = 0.0
        if observation:
            self._prev_state_sig = self._state_signature(observation.state)

        while observation and not observation.end:
            step += 1
            if step > self.max_steps:
                end_reason = "max_steps"
                break
            if self._input_tokens >= self.max_input_tokens:
                end_reason = "token_budget"
                break

            user_msg = self._build_user_message(observation.state, step, last_result)
            self._messages.append({"role": "user", "content": user_msg})

            try:
                tool_name, tool_args, reasoning = self._llm_step()
            except Exception as e:
                self.logger.error(f"LLM call failed at step {step}: {e}")
                end_reason = "llm_error"
                break

            if not tool_name:
                # Smaller models occasionally reply with plain text instead of
                # calling a tool. Nudge them back to tool use rather than aborting.
                last_result = (
                    "No tool call detected. You MUST respond by calling one of "
                    "the available tools (scan_network, find_services, "
                    "exploit_service, find_data, exfiltrate_data). Plain-text "
                    "replies are ignored."
                )
                self._messages.append({"role": "user", "content": last_result})
                transcript.append(
                    {"step": step, "tool": None, "args": None, "result": "no_tool_call"}
                )
                if verbose:
                    print(f"[Step {step}] no tool call (reasoning: {reasoning[:120]!r})")
                continue

            action, parse_error = self.parse_tool_call(
                tool_name, tool_args or {}, observation.state
            )
            tool_call_id = self._messages[-1]["tool_calls"][0]["id"]

            if action is None:
                last_result = (
                    f"INVALID tool call {tool_name}({tool_args}): {parse_error}"
                )
                self._append_tool_result(tool_call_id, last_result)
                if self.action_memory_size > 0:
                    self._action_memory.append(
                        (step, f"{tool_name}({tool_args})", "invalid")
                    )
                transcript.append(
                    {
                        "step": step,
                        "tool": tool_name,
                        "args": tool_args,
                        "result": "invalid",
                    }
                )
                if verbose:
                    print(f"[Step {step}] INVALID tool call: {tool_name}({tool_args})")
                continue

            if verbose:
                print(f"[Step {step}] {action.type.name}({action.parameters})")

            observation = self.make_step(action)
            observation = filter_log_files_from_state(observation)
            if observation and observation.reward is not None:
                total_reward += float(observation.reward)
            last_result = (
                f"{action.type.name} executed. reward={observation.reward}. "
                "See updated state below."
            )
            self._append_tool_result(tool_call_id, last_result)

            if self.action_memory_size > 0 and observation:
                new_sig = self._state_signature(observation.state)
                evaluation = (
                    "helpful" if new_sig != self._prev_state_sig else "unhelpful"
                )
                self._action_memory.append(
                    (step, f"{action.type.name}({action.parameters})", evaluation)
                )
                self._prev_state_sig = new_sig
            transcript.append(
                {
                    "step": step,
                    "tool": tool_name,
                    "args": tool_args,
                    "reasoning": reasoning,
                    "reward": observation.reward,
                }
            )

        if observation and observation.end:
            end_reason = (
                observation.info.get("end_reason", "game_end").value
                if observation.info
                and hasattr(observation.info.get("end_reason"), "value")
                else str(
                    observation.info.get("end_reason", "game_end")
                    if observation.info
                    else "game_end"
                )
            )

        won = bool(
            observation
            and observation.info
            and observation.info.get("end_reason") == AgentStatus.Success
        )
        detected = bool(
            observation
            and observation.info
            and observation.info.get("end_reason") == AgentStatus.Fail
        )

        return EpisodeOutcome(
            won=won,
            detected=detected,
            steps=step,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            end_reason=end_reason,
            total_reward=total_reward,
            transcript=transcript,
        )


def main():
    parser = argparse.ArgumentParser(description="Raw ReAct baseline for NetSecGame")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument(
        "--base_url",
        default=None,
        help="OpenAI-compatible endpoint URL (e.g. http://localhost:8000/v1 for vLLM/Ollama).",
    )
    parser.add_argument("--mission", default=None)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--max_input_tokens", type=int, default=200_000)
    parser.add_argument("--max_output_tokens_per_call", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--show_valid_actions",
        choices=["none", "summary", "full"],
        default="none",
        help=(
            "Action-list scaffolding shown to the LLM each step. "
            "'none' = pure ReAct (default), state + tool schemas only. "
            "'summary' = counts per ActionType, O(1) in size. "
            "'full' = enumerate up to --valid_actions_limit valid actions."
        ),
    )
    parser.add_argument(
        "--valid_actions_limit",
        type=int,
        default=30,
        help="Cap for --show_valid_actions=full.",
    )
    parser.add_argument(
        "--randomize_topology",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomize topology on initial reset and between episodes (default: True).",
    )
    parser.add_argument(
        "--action_memory",
        type=int,
        default=0,
        help=(
            "If > 0, inject a sliding window of the last N actions with intrinsic "
            "evaluation (helpful/unhelpful/invalid) into each user turn. Useful for "
            "smaller models that hallucinate or repeat dead-end actions. Default 0 (off)."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    agent = ReActAgent(
        host=args.host,
        port=args.port,
        role=AgentRole.Attacker,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        mission=args.mission,
        max_steps=args.max_steps,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens_per_call=args.max_output_tokens_per_call,
        temperature=args.temperature,
        show_valid_actions=args.show_valid_actions,
        valid_actions_limit=args.valid_actions_limit,
        action_memory_size=args.action_memory,
    )

    try:
        observation = agent.register()
        # Reset immediately after registration so episode 1 also gets a freshly
        # randomized topology (rather than the server's default starting state).
        observation = agent.request_game_reset(
            randomize_topology=args.randomize_topology,
            seed=421,  # Fixed seed for initial reset to ensure consistent starting state.
        )
        observation = filter_log_files_from_state(observation)
    except Exception as e:
        print(f"Failed to register to NetSecGame at {args.host}:{args.port}")
        print(f"Error: {e}")
        return

    stats = RunStats()
    try:
        for ep in range(1, args.episodes + 1):
            print(f"\n=== Episode {ep} ===")
            outcome = agent.run_episode(observation, verbose=args.verbose)
            total_tok = outcome.input_tokens + outcome.output_tokens
            print(
                f"  result: {'WIN' if outcome.won else ('DETECTED' if outcome.detected else 'FAIL')} "
                f"({outcome.end_reason})"
            )
            print(
                f"  steps: {outcome.steps}, "
                f"reward: {outcome.total_reward:.2f}, "
                f"tokens: {total_tok:,} "
                f"(in={outcome.input_tokens:,} out={outcome.output_tokens:,})"
            )
            stats.record_outcome(outcome)
            if ep < args.episodes:
                observation = agent.request_game_reset(
                    randomize_topology=args.randomize_topology,
                    seed=ep + 2,
                )
                observation = filter_log_files_from_state(observation)
    finally:
        agent.terminate_connection()

    print()
    stats.print_summary()


if __name__ == "__main__":
    main()
