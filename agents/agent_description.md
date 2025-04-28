# AI Agents Description

## Attackers
### Random agent
Random agent is a simple baseline agent that selects action randomly in each step from aviable valid actions with uniform probability. This random agent is primarly used as a baseline in comparison with other agents and to evaluate the complexity of the scenario, performance of the defenders and other comparisons. For reproducibility, it is recommended to use fix random seed when using this agent.
### Interactive agent
The interactive agent primary use is to allow human users to play the game. It provides either CLI or web interface which visualize the state of the game There are several models of operation of this agent:
- Human, without autocompletion of fields nor assistance.
- Human, with autocompletion of fields, but without assistance.
- Human, with autocompletion of fields and LLM assitance.

The autocompletion provides list of available *valid* actions. The LLM assistance uses external LLM to suggest next action in the curret game state.
### Q-learning agent
A **Q-learning agent** learns to act in an environment by **estimating the quality (Q-value)** of taking a certain action in a certain state.

- It keeps a **Q-table**: a lookup table where each entry \( Q(s, a) \) stores the agent's estimate of the **expected cumulative reward** from state \( s \) after taking action \( a \).
- The agent **updates** the Q-values after each interaction with the environment, using the formula:

\[
Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)
\]

where:
- \( s \) = current state
- \( a \) = action taken
- \( r \) = reward received
- \( s' \) = next state
- \( a' \) = possible next action
- \( \alpha \) = learning rate
- \( \gamma \) = discount factor for future rewards

- **Policy**: The agent chooses actions based on the Q-values (using an **epsilon-greedy** strategy: mostly pick the best action, but sometimes explore randomly).
### Sarsa agent
 
A **SARSA agent** (State-Action-Reward-State-Action) learns to act in an environment by **updating the value (Q-value)** of a state-action pair based on the action it *actually* takes, not the best possible action.

- It keeps a **Q-table**: a lookup table where each entry \( Q(s, a) \) stores the agent's estimate of the **expected cumulative reward** from state \( s \) after taking action \( a \).
- The agent **updates** the Q-values after each interaction with the environment, using the formula:

\[
Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma Q(s', a') - Q(s, a) \right)
\]

where:
- \( s \) = current state
- \( a \) = action taken
- \( r \) = reward received
- \( s' \) = next state
- \( a' \) = next action actually taken (according to the policy)
- \( \alpha \) = learning rate
- \( \gamma \) = discount factor for future rewards

- **Policy**: The agent follows a policy like **epsilon-greedy**, selecting actions based on a mix of exploration and exploitation.

A **SARSA agent** learns the value of the actions it actually takes under its policy, making it an **on-policy** method, unlike Q-learning which is **off-policy**.
### LLM agent
## Defenders

## Benign