# NetSecGameAgents
Agents located in this repository should be use in the [Network Security Game](https://github.com/stratosphereips/NetSecGame) environment. They are intented for a navigation and problem solving in the adversarial network-security based environment where they play role of attackers or defenders.

## BaseAgent
All future agents should extend BaseAgent - a minimal implementation of agent capable of interaction with he environment. The base agent also implement logging capabilities for the agent via the `logging` module of python. The logger can be accessed by property `logger`.

For creating an instance of a `BaseAgent`, three parameters have to be used:
1. `host:str` - URL where the game server runs
2. `port: int` - port number where  game server runs
3. `role: str` - Intended role of the agent. Options are `Attacker`, `Defender`, `Human`

When extending the `BaseAgent`, these args should be passed to in the constructor by calling:
```super().__init__(host, port, role)```

There are 4 important methods to be used for interaction with the environment:

1. `register()`: Should be used ONCE in the beginning of the interaction to register the agent in the game. 
    - Uses the class name and `role` specified in the initialization for the registration in the game
    - returns `Observation` which contains the status of the registration and the initial `GameState` if the registration was successful
2. `make_step(Action: action)`: Used for sending a `Action` object to be used as a next step of the agent. Returns `Observation` with new state fo the environment after the action was applied.
3. `request_game_reset()`: Used to RESET the state of the environment to its initial position (e.g. at the end of an episode). Returns `Observation` with state of the environment.
4. `terminate_connection()`: Should be used ONCE at the end of the interaction to properly disconnect the agent from the game server. 

Examples of agents extending the BaseAgent can be found in:
- [RandomAgent](./agents/random/random_agent.py)
- [InteractiveAgent](./agents/interactive_tui/interactive_tui.py)
- [Q-learningAgent](./agents/q_learning/q_agent.py) (Documentation [here](./docs/q-learning.md))

## Agents' compatibility with the environment

| Agent | NetSecGame branch | Tag|
| ----- |-----| ---- |
|[BaseAgent](./agents/base_agent.py) | [main](https://github.com/stratosphereips/NetSecGame/tree/main) | `HEAD`|
|[RandomAgent](./agents/random/random_agent.py) | [main](https://github.com/stratosphereips/NetSecGame/tree/main) | `HEAD`|
|[InteractiveAgent](./agents/interactive_tui/interactive_tui.py) | [main](https://github.com/stratosphereips/NetSecGame/tree/main) | `HEAD`|
|[Q-learning](./agents/q_learning/q_agent.py) | [main](https://github.com/stratosphereips/NetSecGame/tree/main) | `HEAD`|
|[LLM](./agents/llm/llm_agent.py)| [main](https://github.com/stratosphereips/NetSecGame/tree/main) | [realease_out_of_the_cage](https://github.com/stratosphereips/NetSecGame/tree/release_out_of_cage)|
|[LLM_QA](./agents/llm_qa/llm_agent_qa.py)| [main](https://github.com/stratosphereips/NetSecGame/tree/main) | [realease_out_of_the_cage](https://github.com/stratosphereips/NetSecGame/tree/release_out_of_cage)|
|[GNN_REINFORCE](./agents/llm_qa/llm_agent_qa.py)| [main](https://github.com/stratosphereips/NetSecGame/tree/main) | [realease_out_of_the_cage](https://github.com/stratosphereips/NetSecGame/tree/release_out_of_cage)|

### Agent utils
Utility functions in [agent_utils.py](./agents/agent_utils.py) can be used by any agent to evaluate a `GameState`, generate set of valid `Actions` in a `GameState` etc. 

## About us
This code was developed at the [Stratosphere Laboratory at the Czech Technical University in Prague](https://www.stratosphereips.org/).

# To see the results of mlflow

## Locally

1. export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
2. Then run the agent code

From the folder that you run the python
```bash
mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db
```
