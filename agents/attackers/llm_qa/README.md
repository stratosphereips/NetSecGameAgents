# NetSecGame LLM REACT+ Agent

This directory contains an attacker agent for NetSecGame that relies on a
Large Language Model (LLM) to plan actions using the ReAct technique and a
rolling memory of past steps.

## Supported models

The agent can interact with any OpenAI model or an OpenAI-compatible endpoint
such as [Ollama](https://ollama.ai/).

- GPT-4
- GPT-4-turbo
- GPT-3.5-turbo
- GPT-3.5-turbo-16k
- GPT4o / GPT4o-mini
- Any local instruction model exposed through an OpenAI-compatible API

## Features

- Unified `LLMClient` for OpenAI or custom `--base_url` (Ollama compatible)
- Optional tracing with Langfuse via `--enable_tracing` (fallback a no‑op si no está disponible)
- MLflow integration with resilient defaults (local file store por defecto)
- ReAct prompting con memoria configurable y utilidades de reflexión/consistencia
- Modo `--verbose` para imprimir paso a paso lo que hace el agente
- Todas las interacciones (prompts, respuestas, evaluaciones) se guardan en JSON

## Recent changes

- Base class: `LLMActionPlannerBase` expone utilidades genéricas (prompts, `llm_query`, parsing, memoria) para heredar fácilmente.
- Implementación ReAct: `LLMActionPlanner` ahora hereda de la base.
- Paquetización: imports relativos con fallback y `__init__.py` para ejecutar como módulo o script.
- Cliente LLM: soporte OpenAI‑compatible (incl. Ollama). Si no hay `OPENAI_API_KEY`, usa `api_key="ollama"` y normaliza `--base_url` a `.../v1`.
- Verbose: nuevo flag `--verbose` con impresiones de planificación, acción propuesta, ejecución y resultado por paso.
- MLflow: por defecto `--mlflow_tracking_uri file:./mlruns`, `--mlflow_timeout` y fallback para deshabilitar si el servidor no está disponible.
- Langfuse: `--enable_tracing` activa tracer; se ajustó para no enviar `parent_span` (el SDK maneja el contexto automáticamente).

## Prerequisites

- Python 3.12+
- A running instance of the NetSecGame server
- Optional MLflow tracking server
- Access to an OpenAI API key or a local OpenAI-compatible endpoint

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Run as a module (recommended):

```bash
python -m NetSecGameAgents.agents.attackers.llm_qa.llm_agent_qa \
  --llm qwen:4b --test_episodes 2 --verbose
```

Run against a local Ollama endpoint (OpenAI‑compatible):

```bash
python llm_agent_qa.py --llm qwen:4b --test_episodes 2 \
  --base_url http://localhost:11434 --verbose
# Nota: el agente normaliza a http://localhost:11434/v1 automáticamente
```

Enable Langfuse tracing (set creds in .env):

```bash
python llm_agent_qa.py --llm gpt-4 --enable_tracing --verbose
```

### Arguments

| Argument           | Description                                | Default     |
|--------------------|--------------------------------------------|-------------|
| `--llm`            | LLM model to use                           | `gpt-3.5-turbo`|
| `--test_episodes`  | Number of test episodes to run             | `30`        |
| `--memory_buffer`  | Number of past actions to remember         | `5`         |
| `--host`           | Host address of the NetSecGame server      | `127.0.0.1` |
| `--port`           | Port number of the server                  | `9000`      |
| `--base_url`       | Base URL for OpenAI-compatible APIs        | `None`      |
| `--enable_tracing` | Enable Langfuse tracing if available       | `False`     |
| `--disable_mlflow` | Disable MLflow logging                     | `False`     |
| `--mlflow_tracking_uri` | MLflow URI (`file:./mlruns` por defecto) | `file:./mlruns` |
| `--mlflow_timeout` | Timeout (s) para peticiones MLflow         | `3`         |
| `--verbose`        | Imprime progreso paso a paso               | `False`     |

## Output and Logging

- Console: con `--verbose` verás por paso planificación, acción propuesta/validez, efecto y resultado (reward/end).
- File log: se escribe en `llm_react.log` (independiente del `--verbose`).
- JSON: prompts/respuestas/evaluaciones por episodio en `episode_data.json`.
- MLflow: métricas (win rate, detections, returns, steps, promedio) al tracking URI configurado.

Start the MLflow UI to inspect results:

```bash
mlflow ui
```

Then visit [http://localhost:5000](http://localhost:5000) in your browser.

## Langfuse setup (optional)

Define en `.env` de este directorio:

```
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com  # o tu host self‑hosted
# opcional
LANGFUSE_DEBUG=1
```

Actívalo con `--enable_tracing`. Si el SDK/credenciales no están, el tracer cae en no‑op.

## How it works

1. The agent registers with the game server.
2. For each step, the environment state and memory are sent to the LLM.
3. The LLM returns a JSON-formatted action plan.
4. The environment executes the action and provides feedback.
5. Feedback is stored and used for future decisions via a rolling memory
   window.
6. Prompts are defined in `prompts.yaml`.

Extensibility:

- Base class para herencia: `LLMActionPlannerBase` en `llm_action_planner_base.py` expone métodos reutilizables:
  - `llm_query(...)`, `parse_response(...)`, `create_mem_prompt(...)`, `remove_reasoning(...)`, self‑consistency y reflection opcional
- Implementa tu planner heredando y definiendo tu ciclo de prompts/etapas (p.ej. otro protocolo distinto a ReAct).


