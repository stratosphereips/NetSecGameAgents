# Random Agent
Random agent is a simple baseline agent that selects action randomly in each step from aviable *valid* actions with uniform probability.

The random agent is primarly used as a baseline in comparison with other agents and to evaluate the complexity of the scenario.

## Installation
To install the random agent, follow the installation guide in the NetSecGameAgents with `[random]` option:

```
pip install -e .[random]
```
It is recommended to install the agent in a virtual environment.

## Running the agent
The agent can be run with following command:
```
python3 -m agents.attackers.random.random_agent 
```
