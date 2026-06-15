# Q-Learning Agent
This agent implements the Q learning algorithm

## Installation
To install the random agent, follow the installation guide in the NetSecGameAgents with `[q_learning]` option:

```
pip install -e .[q_learning]
```
It is recommended to install the agent in a virtual environment.

## Running the agent
The agent can be run with following command:
```
python3 -m agents.attackers.q_learning.q_agent
```

## Terminal States and Time-Limit Truncation

The agent distinguishes true terminal outcomes from episodes stopped by the
maximum-step limit:

- `AgentStatus.Success` and `AgentStatus.Fail` are terminal outcomes. There is
  no future action after either outcome, so the Q-learning target is only the
  final reward:

  ```text
  target = reward
  ```

- `AgentStatus.TimeoutReached` is a time-limit truncation. The step limit
  restricts the length of a training episode, but does not represent a terminal
  state of the underlying task. The update therefore continues to bootstrap
  from the final observed state:

  ```text
  target = reward + gamma * max_a Q(next_state, a)
  ```

Treating a training time limit as a terminal state would incorrectly force the
estimated future value at the cutoff to zero. This handling follows the
time-limit treatment described by Pardo et al., "Time Limits in Reinforcement
Learning" (2018): https://proceedings.mlr.press/v80/pardo18a.html
