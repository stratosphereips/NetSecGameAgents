# Q-Learning Agent

The q-learning agent uses the traditional q-learning algorithm to learn how to play the network security game.

The characteristics of this agent are:

## Epsilon Decay
It uses an epsilon decay function to decrase the value of epsilon from a parameter `epsilon_start` to an `epsilon_end` for `epsilon_max_episodes` episodes. The function is :
    
  
  ```math
  decay_rate = max((epsilon\_max\_episodes - episode\_num) / epsilon\_max\_episodes, 0)
  ```

  ```math
  current\_epsilon = (epsilon\_start - epsilon\_end) * decay\_rate + epsilon\_end
  ```

This is a simple linear interpolation.

## E-greedy
The e-greedy algorithm breaks ties when more than one state-actions pairs have the same value, using a random function.

## Training
The model can be read from file and store to a file every time. Just use the parameter `--previous_model` to load a specific model.

## Testing
You can specify the total amount of episodes, also every how many episodes you want to evaluate the model, and on each evaluation, for how many episodes you want to test. A small diagram showing this is

![](https://github.com/stratosphereips/NetSecGameAgents/blob/q-learning-improve/docs/training-testing-diagram.png)

During testing the qtable is not updated.

## Storing actions
You can use the parameter `--store_actions` to store the actions, state, reward, end and info to a file for later checking. Be careful that if you do this during training or a long testing it can get very large.

## MlFlow
It uses mlflow to log the experiments and you can say which Experiment ID it is. Automatically will store the commit hash of the NetSecEnvAgents repository and the NetSecEnv repository.

Now we send to mlflow
- The parameters used for the python script
- The experiment name and id
- For each run of training, many metrics.
- For each run of testing, many metrics.
- The configuration file of the environment used, so to remember which game we played.
- The file of the configuration of the environment used in the game as artifact. So we know the exact parameters of the game played. This file belongs to the env, so it may not be available to the agent. This file is only used here as documentation.

## Check of q-table
There is a small python file, called `check_q_table.py` that can be used to check the content of the q-table to see if the values make sense. It can print many states, all actions or the top one, in colors.

## Inner reward
The qlearning agent has its own recomputation of inner reward. This means that it assigns special rewards to certain events to help learning. The current ones are:
  - -1 for the normal step
  - -100 for a time out (max steps reached)
  - 1000 for the goal
  - -1000 for detection

## Actions Choosing
All agents choose which actions can be done on each state, so only the possible actions are selected. This allows for a 'continually' growing q-table without the need to know all the possible actions in advance. Now we even filter that an action in a private IP or Net can only be done from a private IP or Net. So no trying to scan ports in a local private IP from a public one (which would never work).

## Conceptual Agent
In the current version the qlearning agent implements an early stage of conceptual learning. We implemented a function that translates the observation comming from the env to concepts that are general and not dependent on the IPv4 Network octets. So now it doesnt matter the exact network where the agent is playing, it will still learn to win.

For now we only translate Netorks and not IPs.

Regarding the hosts, they are separated in six categories:
1. web: web services
2. terminal: remote access and administration and terminals
3. db: databases
4. files: files access
5. external: not in the local network
6. unknown: unknown

If a host does not have a service yet, then it goes into the 'unknown'. 
After a host is added to a group that is not 'unknown', then it is taken out of 'unknown'.

