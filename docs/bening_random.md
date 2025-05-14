# Bening Random Agent
The gening random agent is an agent designed to do bening actions in a random order.

Actions are the same as in the rest of the agents, but there should not be a specific goal for winning the game, but more like the actions following other goal, such as 'working'.
This agent in particular implements random actions, which in theory looks very similar to the `random agent`. The only differences is that the `bening random agent` has a limitation of APM, or Actions Per Minute, since it is supposed to mimic a human operator.

The other differene is that the `bening random agent` does not do the actions `ScanNetwork`, `ScanServices` or `ExploitServices`. However, it does do the action `FindServices` because it is possible that a network administrator is actually scanning for ports to see which ones are open. 

## False Positives
If any action done by any benign agent is blocked by the FW of the game, then it is counted as a false positive and the final reward of the defender is decreased a fixed amount per false positive.