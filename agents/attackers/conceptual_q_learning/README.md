# Conceptual Attacker Agent

The conceptual attacker agent is a modification to the Q-learning attacker to avoid depending on IP addresses to play the game, and instead convert each IP address into a concept, just as humans do when they attack a network.

# Install
Install the dependencies of this agent with 

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -e ..
python -m pip install -e ".[conceptual_q_learning]"
```

# Run the Agent
If the NetSecGame server is running in localhost, port 9000/TCP, then:

```
python -m agents.attackers.conceptual_q_learning.conceptual_q_agent --host localhost --port 9000 --episodes 1 --experiment_id test-1 --env_conf ../AIDojoCoordinator/netsecenv_conf.yaml
```

To record testing and evaluation trajectories in the `TrajectoryRecorder` JSONL
format, add:

```bash
--record_trajectories True --trajectoriesdir /path/to/trajectories
```
