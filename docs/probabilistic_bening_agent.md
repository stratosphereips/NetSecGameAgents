# Bening Agent
The benign agent is an agent designed to do benign actions following some model.

Actions are the same as in the rest of the agents, but there should not follow a specific goal for *winning* the game, but more like the actions following other goal, such as 'working'.
This agent in particular implements actions that follow probability distributions. This agent has a limitation of APM, or Actions Per Minute, since it is supposed to mimic a human operator.

The other difference is that the `bening agent` does not do the actions `ScanNetwork`, and `ExploitServices`. However, it does do the action `FindServices` because it is possible that a network administrator is actually scanning for ports to see which ones are open. The total list of actions that it can do is `FindServices`, `ExfiltrateData` and `FindData`.

This agent works like this.
1. There is a configuration file that defines how the agent behaves.
2. The behavior is a set of actions




- Check email
- Browse the internet
- Check internal email
- Access DB
- Work (means work internally in the computer, so just default flows are sent)
- Work (means work internally in the computer, so just default flows are sent)
- Work (means work internally in the computer, so just default flows are sent)
- Check social networks
- Check security of server A