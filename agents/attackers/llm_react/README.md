# NetSecGame LLM REACT+ Agent

This repository provides an implementation of an attacker agent for NetSecGame controlled using a Large Language Model (LLM). The agent uses ReAct-style prompting and memory to plan and take actions in a simulated environment.

## Supported models


Supported models include OpenAI's GPT family and others local models using Ollama API

- GPT4 (OpenAI)
- GPT-4-turbo (OpenAI)
- GPT-3.5-turbo (OpenAI)
- GPT-3.5-turbo-16k (OpenAI)
- GPT4o-mini (OpenAI)
- GPT4o (OpenAI)
- Any local instruction model compatible with the OpenAI API should work, however there is no guarantee it can solve any scenario.

## Features

- Uses LLMs to control an agent in a simulated cybersecurity game
- Supports OpenAI and Hugging Face APIs
- Collects detailed metrics with MLflow
- Tracks prompts, LLM responses, and evaluation data
- Saves all interaction data in JSON format
- Allows configurable memory buffer for multi-step planning

## Prerequisites

- Python 3.12+
- A running instance of the NetSecGame server
- MLflow tracking server (optional but recommended)
- Access to OpenAI API, a locally hosted Ollama API or any other OpenAI compatible API.

Install required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the agent with a local model

```bash
python llm_agent_qa.py --llm llama3.1:8b --test_episodes 20 --memory_buffer 5 --api_url http://localhost:11434/v1/
```

### Arguments

| Argument          | Description                                | Default                       |
|-------------------|--------------------------------------------|-------------------------------|
| `--llm`           | LLM model to use (e.g.gpt-4,lama3.1`)      | `gpt4o-mini`        		 |
| `--test_episodes` | Number of test episodes to run             | `30`                          |
| `--memory_buffer` | Number of past actions to remember         | `5`                           |
| `--host`          | Host address of the NetSecGame server      | `127.0.0.1`                   |
| `--port`          | Port number of the server                  | `9000`                        |
| `--api_url`       | Endpoint for OpenAI or Ollama model API    | `http://127.0.0.1:11434/v1/`  |
| `-disable_mlflow` | Disable mlflow logging			 | `False`  			 |


## Output and Logging

- Metrics like win rate, detection rate, and returns are logged to MLflow
- All prompts, responses, and evaluations are stored in `episode_data.json`
- Execution logs are written to `llm_qa.log`

To view experiment results, start the MLflow UI:

```bash
mlflow ui
```

Then visit [http://localhost:5000](http://localhost:5000) in your browser.

## How It Works

1. The agent registers with the game server.
2. At each step, the environment state and past memory are passed to the LLM.
3. The LLM returns a JSON-formatted action plan.
4. The environment executes the action and provides feedback.
5. Feedback is stored and used to inform future steps via a rolling memory window.
6. Prompts for the agents are located in `prompts.yaml`


## License

This project is licensed under the MIT License.

# Publications

[1] Rigaki, M., Lukáš, O., Catania, C., & Garcia, S. (2024). Out of the cage: How stochastic parrots win in cyber security environments. In Proceedings of the 16th International Conference on Agents and Artificial Intelligence (pp. 774–781). SCITEPRESS – Science and Technology Publications. [https://doi.org/10.5220/0012391800003636](https://doi.org/10.5220/0012391800003636)

[2] Rigaki, M., Catania, C., & Garcia, S. (2024). Hackphyr: A local fine-tuned LLM agent for network security environments. arXiv. [https://arxiv.org/abs/2409.11276](https://arxiv.org/abs/2409.11276)
