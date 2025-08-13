# Conceptual Attacker Agent

The conceptual attacker agent is a modification to the Q-learning attacker to avoid depending on IP addresses to play the game, and instead convert each IP address into a concept, just as humans do when they attack a network.


# Run

Example
```
cd NetSecAgents
python -m agents.attackers.conceptual_q_learning.conceptual_q_agent --host localhost --port 9000 --episodes 2000 --experiment_id 50 --env_conf ../AIDojoCoordinator/netsecenv_conf.yaml
```