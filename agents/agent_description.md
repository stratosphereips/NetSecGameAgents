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
There are two types of defender agents currently implemented in AI Dojo. The stochastic defenders simulate a SIEM with full access to the hosts in the network. The primary action of the defenders is `BlockIP` action by which they can restric access to certain hosts. If successful, this leads to preventing the attacker's goals.
### Random defender
Random agent is a simple baseline agent that selects action randomly in each step from aviable valid actions with uniform probability. This random agent is primarly used as a baseline in comparison with other agents and to evaluate the complexity of the scenario, performance of the defenders and other comparisons. For reproducibility, it is recommended to use fix random seed when using this agent.

### Stochastic defender
Stochastic defender makes the blocking based on heuristic and probabiliy distribution over the action types. The agent repeadly checks the logs in each host by using `FindData` action and analyzing the content. This agent implements a detection heuristic that only applies probabilistic detection after certain suspicious patterns have been observed in an agent’s behavior. It considers the most recent actions in the log within a **fixed-size time window** (default size is 5) to analyze short-term patterns.

The agent checks for three types of suspicious behavior: (1) a high ratio of the same action type in the recent window, (2) multiple consecutive occurrences of the same action type, and (3) frequent repetition of the exact same parametrized action across the entire episode. For each check, it compares the observed statistics against predefined thresholds that differ for each action type. For instance, some actions are flagged if their proportion in the window is high, while others are flagged if they are repeated too many times consecutively or across the whole episode.

If an action type is associated with a consecutive threshold (like `ScanNetwork`, `FindServices`, or `ExfiltrateData`), the detection logic requires that either the proportion of the action in the window exceeds its threshold, or the maximum number of consecutive repetitions crosses its consecutive threshold. If either condition is met, the method then applies a stochastic detection (i.e., it randomly detects the action based on a fixed probability assigned to the ActionType). If neither condition is met, detection does not occur.

On the other hand, if an action type is associated with an episode-wide repetition threshold (like `ExploitService` or `FindData`), detection is triggered if either the action’s proportion in the recent window is too high, or the total number of repetitions throughout the episode exceeds its threshold. Again, if triggered, detection occurs stochastically.

For any other actions not explicitly listed in the threshold dictionaries, detection is automatically disabled — regardless of their frequency or behavior. Additionally, if there are not enough actions yet to fill the time window (i.e., if the episode is still short), the method will simply not attempt detection and return False.

### SLIPS defender
SLIPS defender is based on open-source ML-drived IDS called [Stratoshpere Linux IPS](https://github.com/stratosphereips/StratosphereLinuxIPS). It is using wide range of modules to detect suspicious behavior in the network traffic. SLIPS agent is directly connected to CYST simulation engine as it operates on Netflows, not the high-level GameState representation.

## Benign
The purpose of the benign agents in the AI Dojo is to create realistic environment for the simulation. In reality, there is a majority of normal users and their activity in which the attackers might hide. Thus, to properly train and evaluate the defenders' capabilities, this normal, bening activity has to be included.

### Random benign agent
Random benign agent is a limited version of the attacker. It can only perform a subset of action (Finding hosts and data and moving data). The agent has a APM limit which can be modified when starting.