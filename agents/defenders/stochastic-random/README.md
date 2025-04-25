# Stochastic Random Defender

The stochastic random defender is an agent that tries to defend the network by randomly blocking other computers in the firewall of the network. The decision to block or not is based on a uniform random probability distribution on the available known hosts on each step. But since the agent starts with perfect knowledge of the network, then any host can be blocked at any time.

# How it works
1. Play as a defender agent only with one action avilable: BlockIP()
2. Use a model to decide which actions to do. The model can be trained in any way.
3. The defender starts knowing all the networks, hosts, services and data, and controlling all the hosts.
4. Aidojo adds every action taken by each agent to a special file calleed 'actions.txt' with the description being `{}`
5. However, since that is the beginning of the game, the data does not have content from action logs.
6. Therefore the agent can decide to `find` the data in any host any time it needs and then it will `update` its view on that data.