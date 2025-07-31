# NetSecGameAgents
Agents located in this repository should be used in the [Network Security Game](https://github.com/stratosphereips/NetSecGame) environment. They are intended for navigation and problem solving in the adversarial network-security based environment where they play the role of attackers or defenders.

## Installation
Agents need their own set of libraries which are installed separatedly from the AiDojo environment.

To run an agent you need to install
- The library of the AIDojoCoordinator
- The libraries needed by your agent

We recommend to use virtual environment when installing.

```bash
python -m venv aidojo-agents
```

To activat the venv, run:
```
source aidojo-agents/bin/activate
```

Be sure you are in the directory of this _NetSecGameAgents_ repository.

### Install the libraries of the AiDojoCoordinator
Agents requires components of the [NeSecGame](https://github.com/stratosphereips/NetSecGame) to run properly so make sure it is installed first.
The code for NetSecGame is assumed to be in the previous directory

- `python -m pip install -e ..`

To install the required packages for each agent, you can run 
```
python -m pip install -e .[<name-of-the-agent>] 
```

For example `python -m pip install -e ".[tui,llm]"`

For a complete list of agents to install the dependencies see the pyproject.toml file.


## Runing the agent
To run the agents, use
```
python3 -m <path-to-the-agent>
```
For example, to run the random attackers:
```
python3 -m agents.attackers.random.random_agent
```
## BaseAgent
All future agents should extend BaseAgent - a minimal implementation of agent capable of interaction with the environment. The base agent also implements logging capabilities for the agent via the `logging` python module. The logger can be accessed by property `logger`.

For creating an instance of a `BaseAgent`, three parameters have to be used:
1. `host: str` - URL where the game server runs
2. `port: int` - port number where game server runs
3. `role: str` - Intended role of the agent. Options are `Attacker`, `Defender`, `Human`

When extending the `BaseAgent`, these args should be passed to the constructor by calling:
```
super().__init__(host, port, role)
```

There are 4 important methods to be used for interaction with the environment:

1. `register()`: Should be used ONCE at the beginning of the interaction to register the agent in the game. 
    - Uses the class name and `role` specified in the initialization for the registration in the game
    - returns `Observation` which contains the status of the registration and the initial `GameState` if the registration was successful
2. `make_step(Action: action)`: Used for sending an `Action` object to be used as a next step of the agent. Returns `Observation` with new state of the environment after the action was applied.
3. `request_game_reset()`: Used to RESET the state of the environment to its initial position (e.g. at the end of an episode). Returns `Observation` with state of the environment.
4. `terminate_connection()`: Should be used ONCE at the end of the interaction to properly disconnect the agent from the game server. 

Examples of agents extending the BaseAgent can be found in:
- [RandomAgent](./agents/attackers/random/random_agent.py)
- [InteractiveAgent](./agents/attackers/interactive_tui/interactive_tui.py)
- [Q-learningAgent](./agents/attackers/q_learning/q_agent.py) (Documentation [here](./docs/q-learning.md))

## Agent's types
There are three types of roles an agent can play in NetSecEnv:
1. Attacker
2. Defender
3. Benign

Agents of each type are stored in the corresponding directory within this repository:
```
├── agents
    ├── attackers
        ├── concepts_q_learning
        ├── double_q_learning
        ├── gnn_reinforce
        ├── interactive_tui
        ├── ...
    ├── defenders
        ├── random
        ├── probabilistic
    ├── benign
        ├── benign_random
```

### Agent utils
Utility functions in [`agent_utils.py`](./agents/agent_utils.py) can be used by any agent to evaluate a `GameState`, and generate a set of valid `Actions` in a `GameState`, etc. 
Additionally, there are several files with utils functions that can be used by any agents:
- [`agent_utils.py`](./agents/agent_utils.py) Formatting GameState and generation of valid actions
- [`graph_agent_utils.py`](./agents/graph_agent_utils.py): GameState -> graph conversion
- [`llm_utils.py`](./agents/llm_utils.py): utility functions for LLM-based agents

## Agents' compatibility with the environment

| Agent | NetSecGame branch | Tag| Status |
| ----- |-----| ---- | ---- |
|[BaseAgent](./agents/base_agent.py) | [main](https://github.com/stratosphereips/NetSecGame/tree/main) | `HEAD`| ✅ |
|[Random Attacker](./agents/random/random_agent.py) | [main](https://github.com/stratosphereips/NetSecGame/tree/main) | `HEAD`| ✅ |
|[InteractiveAgent](./agents/interactive_tui/interactive_tui.py) | [main](https://github.com/stratosphereips/NetSecGame/tree/main) | `HEAD`| ✅ |
|[Q-learning](./agents/q_learning/q_agent.py) | [main](https://github.com/stratosphereips/NetSecGame/tree/main) | `HEAD`| ✅ |
|[LLM](./agents/llm/llm_agent.py)| [main](https://github.com/stratosphereips/NetSecGame/tree/main) | [realease_out_of_the_cage](https://github.com/stratosphereips/NetSecGame/tree/release_out_of_cage)| ✅ |
|[LLM_QA](./agents/llm_qa/llm_agent_qa.py)| [main](https://github.com/stratosphereips/NetSecGame/tree/main) | [realease_out_of_the_cage](https://github.com/stratosphereips/NetSecGame/tree/release_out_of_cage)| ✅ |
|[GNN_REINFORCE](./agents/llm_qa/llm_agent_qa.py)| [main](https://github.com/stratosphereips/NetSecGame/tree/main) | [realease_out_of_the_cage](https://github.com/stratosphereips/NetSecGame/tree/release_out_of_cage)| ✅ |
|[Random Defender](./agents/defenders/random/random_agent.py)| [main](https://github.com/stratosphereips/NetSecGame/tree/main) | | 👷🏼‍♀️ |
|[Probabilistic Defender](./agents/defenders/probabilistic/probabilistic_agent.py)| [main](https://github.com/stratosphereips/NetSecGame/tree/main) | | 👷🏼‍♀️ |

## Export to mlflow

Every agent by default exports the experiment details to a local mlflow directory.

If you want to see the local mlflow data do

```bash
pip install mlflow
mlflow ui -p 5001
```

If you want to export the local mlflow to a remote mlflow you can use our util 

```bash
python utils/export_import_mlflow_exp.py --experiment_id 783457873620024898 --run_id 5f2e4a205b7745259a4ddedc12d71a74 --remote_mlflow_url http://127.0.0.1:8000 --mlruns_dir ./mlruns
```

## Install

- create new env
- install numpy
- install coor `pip install -e ..`
- optionally install mlflow

## About us
This code was developed at the [Stratosphere Laboratory at the Czech Technical University in Prague](https://www.stratosphereips.org/) as part of the [AIDojo Project](https://www.stratosphereips.org/ai-dojo).
