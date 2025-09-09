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

- Unified `LLMClient` for OpenAI or custom `--base_url`
- Optional tracing with Langfuse via `--enable_tracing` (falls back to a
  no-op tracer if the SDK or credentials are missing)
- MLflow integration for experiment tracking
- ReAct prompting with configurable memory buffer
- All prompts, responses and evaluations stored as JSON

## Recent changes

- `LLMActionPlanner` now inherits from `LLMActionPlannerBase` for easier
  extension
- Tracing is handled by a small abstraction in `tracer.py`
- Langfuse usage is opt-in with a flag instead of a hard dependency
- The agent accepts `--base_url` to target local endpoints like Ollama

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

Run the agent against a local Ollama model:

```bash
python llm_agent_qa.py --llm llama3.1:8b --test_episodes 20 --memory_buffer 5 \
    --base_url http://localhost:11434/v1/
```

Enable Langfuse tracing (credentials and SDK must be available):

```bash
python llm_agent_qa.py --llm gpt-4 --enable_tracing
```

### Arguments

| Argument           | Description                                | Default     |
|--------------------|--------------------------------------------|-------------|
| `--llm`            | LLM model to use                           | `gpt4o-mini`|
| `--test_episodes`  | Number of test episodes to run             | `30`        |
| `--memory_buffer`  | Number of past actions to remember         | `5`         |
| `--host`           | Host address of the NetSecGame server      | `127.0.0.1` |
| `--port`           | Port number of the server                  | `9000`      |
| `--base_url`       | Base URL for OpenAI-compatible APIs        | `None`      |
| `--enable_tracing` | Enable Langfuse tracing if available       | `False`     |
| `--disable_mlflow` | Disable MLflow logging                     | `False`     |

## Output and Logging

- Metrics like win rate, detection rate and returns are logged to MLflow
- Prompts, responses and evaluations are stored in `episode_data.json`
- Execution logs are written to `llm_qa.log`

Start the MLflow UI to inspect results:

```bash
mlflow ui
```

Then visit [http://localhost:5000](http://localhost:5000) in your browser.

## How it works

1. The agent registers with the game server.
2. For each step, the environment state and memory are sent to the LLM.
3. The LLM returns a JSON-formatted action plan.
4. The environment executes the action and provides feedback.
5. Feedback is stored and used for future decisions via a rolling memory
   window.
6. Prompts are defined in `prompts.yaml`.

## License

This project is licensed under the MIT License.

## Publications

- Rigaki, M., Lukáš, O., Catania, C., & Garcia, S. (2024). *Out of the cage:
  How stochastic parrots win in cyber security environments*. In Proceedings of
  the 16th International Conference on Agents and Artificial Intelligence
  (pp. 774–781). SCITEPRESS.
- Rigaki, M., Catania, C., & Garcia, S. (2024). *Hackphyr: A local fine-tuned
  LLM agent for network security environments*. arXiv. <https://arxiv.org/abs/2409.11276>

