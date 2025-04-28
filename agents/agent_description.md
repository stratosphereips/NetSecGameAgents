# AI Agents Description
The agents in the AI Dojo can play the role of attackers (red team), defenders (blue team) or the benign agents (simulation of normal users).
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

- It keeps a **Q-table**: a lookup table where each entry `Q(s, a)` stores the agent's estimate of the **expected cumulative reward** from state `s` after taking action `a`.
- The agent **updates** the Q-values after each interaction with the environment, using the formula:

`Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') - Q(s, a) ]`

where:
- `s` = current state
- `a` = action taken
- `r` = reward received
- `s'` = next state
- `a'` = possible next action
- `α` = learning rate
- `γ` = discount factor for future rewards

**Policy**: The agent chooses actions based on the Q-values (often using an **epsilon-greedy** strategy: mostly picking the best action, but sometimes exploring randomly).
Detailed description of this agent can be found in [Catch Me If You Can: Improving Adversaries in Cyber-Security With Q-Learning Algorithms](https://arxiv.org/abs/2302.03768)

### Sarsa agent
 
A **SARSA agent** (State-Action-Reward-State-Action) learns to act in an environment by **updating the value (Q-value)** of a state-action pair based on the action it *actually* takes, not the best possible action.

- It keeps a **Q-table**: a lookup table where each entry `Q(s, a)` stores the agent's estimate of the **expected cumulative reward** from state `s` after taking action `a`.
- The agent **updates** the Q-values after each interaction with the environment, using the formula:

`Q(s, a) ← Q(s, a) + α [ r + γ Q(s', a') - Q(s, a) ]`

where:
- `s` = current state
- `a` = action taken
- `r` = reward received
- `s'` = next state
- `a'` = next action actually taken (according to the policy)
- `α` = learning rate
- `γ` = discount factor for future rewards

- **Policy**: The agent follows a policy like **epsilon-greedy**, selecting actions based on a mix of exploration and exploitation.

A **SARSA agent** learns the value of the actions it actually takes under its policy, making it an **on-policy** method, unlike Q-learning which is **off-policy**.
### LLM agent
The LLM agent is based on the large language model. It uses advance prompting techniques to process the textual description of the role of the agent, its goals, the current state of the game and previous actions to generate the next action. Several publicly avaiable language models were tested and can be used with this agent. Details about the agent architecture, the prompts and its performancce can be found in [Out of the Cage How Stochastic Parrots Win in Cyber Security
Environments](https://arxiv.org/pdf/2308.12086)
## Defenders
### Random defender

### Stochastic defender

### Stochastic Defender with thresholds

### Slips defender

## Benign
### Random benign agent