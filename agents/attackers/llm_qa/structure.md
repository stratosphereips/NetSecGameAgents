# `llm_qa` Structure

This directory implements an attacker agent that uses a Large Language Model (LLM) to plan actions in NetSecGame using a ReAct-style loop (reason → decide → act) with memory, response validation, and optional tracing.

## Main Files and Purpose

- `llm_agent_qa.py`: CLI entry point. Orchestrates episodes against the game server, initializes the LLM client and tracer, and constructs the action planner.
- `llm_action_planner_base.py`: Provider-agnostic base class for LLM action planners. Centralizes prompt loading, instruction building, LLM querying with retries, tracing, parsing/validation, and utilities (memory, self-consistency, reasoning stripping).
- `llm_action_planner.py`: Concrete ReAct planner built on the base. Implements a two-stage flow (reasoning and action selection), validates JSON, and converts it to environment actions.
- `llm_client.py`: Minimal wrapper over the `openai` client for OpenAI-compatible endpoints (OpenAI or Ollama via `base_url`).
- `tracer.py`: Tracing abstraction (Noop by default; Langfuse if enabled). Exposes `start_trace`, `start_span`, `start_generation`, and `flush` to instrument LLM calls and agent steps.
- `prompts.yaml`: Instruction templates (Jinja2), CoT examples, and questions (Q1–Q4) used by the planner.
- `validate_responses.py`: Validates the LLM's JSON response per action (required fields, types, nested objects).
- `README.md`: Usage overview, supported models, and the high-level flow.
- `.env`: Optional variables (Langfuse and/or `OPENAI_API_KEY`).
- `episode_data.json` / `llm_react.log`: Execution outputs (prompts/responses per episode and logs).

## Inheritance Design: `LLMActionPlannerBase`

The base class defines the interface and utilities reused by concrete planners, decoupling LLM/tracing “plumbing” from decision logic.

- Initialization and configuration
  - Loads YAML via `ConfigLoader` (searches nearby paths).
  - Builds dynamic instructions with Jinja2 from the episode goal.
  - Manages buffers: `prompts`, `states`, `responses`, and `session_id`.

- LLM querying with retries and tracing
  - `llm_query(...)`: wraps `chat.completions.create(...)` with `tenacity` retries, parameters (`max_tokens`, `temperature`, `response_format`), and `tracer.start_generation(...)` to capture input, output, and token usage when available.

- Response parsing and validation
  - `parse_response(text, state)`: attempts `json.loads`, extracts `action` and `parameters`, then delegates to `create_action_from_response(...)` (in `agents/llm_utils.py`) to validate against the game state and, if valid, build an executable `Action`. Returns `(valid, response_dict, action)`.

- Prompting and robustness utilities
  - `update_instructions(goal)`: regenerates instructions from the YAML template.
  - `create_mem_prompt(memory_list)`: summarizes past actions and their goodness to reduce repetition.
  - `check_repetition(...)`: counts repeated memories to, e.g., adjust temperature.
  - `remove_reasoning(text)`: strips `<think>...</think>` blocks for models that expose chain-of-thought.
  - `get_self_consistent_response(...)`: runs N samples and returns the most frequent response (self-consistency) when enabled.

What concrete planners inherit:
- Access to `self.llm` (OpenAI-compatible client), `self.tracer`, `self.model`, `self.config`, `self.instructions`, buffers `prompts/states/responses`, and `self.session_id`.
- Utility methods listed above, especially `llm_query`, `parse_response`, `remove_reasoning`, `get_self_consistent_response`, and the instruction/memory helpers.

## ReAct Planner: `LLMActionPlanner`

Inherits from `LLMActionPlannerBase` and defines the high-level method:

- `get_action_from_obs_react(observation, memory_buf)`:
  - Stage 1 (reasoning): instructions + status + memory + Q1 → `llm_query` (optionally reflection and/or self-consistency). Can apply `remove_reasoning`.
  - Stage 2 (selection): instructions + status + CoT examples + reasoning + memory + Q4 → `llm_query` with JSON `response_format`. Validates with `validate_responses.py`. If invalid, wraps the error in a controlled JSON object.
  - Parses with `parse_response(...)` (which delegates to `create_action_from_response(...)`).
  - Records inputs/outputs and appends to buffers for later analysis.

This lets the subclass focus on prompting strategy and action selection while reusing the base infrastructure.

## Tracing: `tracer.py`

Lightweight abstraction to instrument the agent without hard coupling to a specific provider.

- Interfaces
  - `ITracer`: defines context managers `start_trace`, `start_span`, `start_generation`, and `flush()`.
  - `_NoopSpan`: no-op object with `.update(...)` so caller code can always invoke `.update` safely.

- Implementations
  - `NoopTracer`: no-op; used by default or when Langfuse is unavailable.
  - `LangfuseTracer`: uses the Langfuse SDK if present. Handles API variations across SDK versions and attempts:
    - `start_trace(...)`: create a root trace and reset any lingering SDK context if needed.
    - `start_span(name, parent_span=...)`: attach child spans to the provided parent or create top-level spans.
    - `start_generation(...)`: specialized span for LLM generations to capture parameters, inputs, outputs, and usage.
    - `flush()`: attempts to flush buffers or gracefully shut down the client.

- Usage in code
  - `llm_agent_qa.py` obtains the tracer via `get_tracer(--enable_tracing)` and wraps episodes, steps, and LLM calls.
  - `llm_action_planner_base.py` instruments each `llm_query` with `start_generation(...)`.

## Other Relevant Components

- `agents/llm_utils.py` (outside this folder):
  - `create_status_from_state(state)`: builds the status prompt.
  - `validate_action_in_state(...)` and `create_action_from_response(...)`: validate/build the environment `Action` from the LLM JSON against the current game state.

## End-to-End Flow

1. `llm_agent_qa.py` connects to the game server, registers the agent, and per episode constructs an `LLMActionPlanner`.
2. The planner (subclass) uses inherited methods to query the LLM, validate, parse, and return an action.
3. `BaseAgent` sends the action and receives a new observation; memory is maintained and the loop continues until termination.
4. Prompts, responses, states, and metrics are recorded; tracing is optionally sent to Langfuse.

## Variables and Dependencies

- `.env` (optional): `OPENAI_API_KEY` for GPT models; `LANGFUSE_*` for tracing.
- `requirements.txt`: `openai`, `langfuse`, `dotenv`, `mlflow`, `transformers` (utilities only; this agent does not load HF models).

## Configuration (.env)

Create a `.env` file in this folder to keep credentials and related settings in one place. The agent loads it via `python-dotenv`.

Recommended entries:

```dotenv
# OpenAI (required when using GPT models)
OPENAI_API_KEY=sk-...
# Optional: only if you use a custom OpenAI-compatible proxy
# OPENAI_BASE_URL=https://your-proxy.example.com/v1

# Langfuse (required only when running with --enable_tracing)
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
# Choose the region where your Langfuse project is hosted:
#   US: https://us.cloud.langfuse.com
#   EU: https://cloud.langfuse.com
LANGFUSE_HOST=https://us.cloud.langfuse.com

# (Optional) Convenience variables for your game server
# Note: the current script expects --host/--port CLI flags. You can still
# store them here and reference them in your shell command.
# GAME_HOST=127.0.0.1
# GAME_PORT=9000
```

Notes:
- The script reads `OPENAI_API_KEY` and (optionally) `OPENAI_BASE_URL`. For official OpenAI, do not set `OPENAI_BASE_URL`.
- Tracing with `--enable_tracing` requires `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST`. Langfuse provides two regional hosts (US and EU) — pick the one matching your project region.
- Host/port for the NetSecGame server are passed via `--host` and `--port`. You may keep them in `.env` for convenience and reference them in the command, e.g.:
  - `python llm_agent_qa.py --llm gpt-4o-mini --host ${GAME_HOST:-127.0.0.1} --port ${GAME_PORT:-9000}`
