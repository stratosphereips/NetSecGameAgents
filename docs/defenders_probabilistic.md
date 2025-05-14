# Probabilistic Defender

The probabilistic defender starts with all the hosts controlled, all the hosts known, all the services known and all the data known. Then it picks a host to check the logs from based on a probability distribution, read the logs, and applies a probabilisitic detection based on the identified actions. This implies that the agent can recognize actions based on the logs. Then for each action it blocks the IP doing the action with some probability in the host receiving the action.

There are many things to try in the future, such as:
- Also block in the IPs that are not attacked, to stop future attacks.
- Block in the src host that is doing the action too.

## False Positives
If any action done by any benign agent is blocked by the FW of the game, then it is counted as a false positive and the final reward of the defender is decreased a fixed amount per false positive.