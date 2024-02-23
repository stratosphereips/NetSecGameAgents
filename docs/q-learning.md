# Q-Learning Agent

The q-learning agent uses the traditional q-learning algorithm to learn how to play the network security game.

The characteristics of this agent are:

## Epsilon Decay
It uses an epsilon decay function to decrase the value of epsilon from a parameter `epsilon_start` to an `epsilon_end` for `epsilon_max_episodes` episodes. The function is :

\[ \text{{decay\_rate}} = \max\left(\frac{{\text{{epsilon\_max\_episodes}} - \text{{episode\_num}}}}{{\text{{epsilon\_max\_episodes}}}}, 0\right) \]

\[ \text{{current\_epsilon}} = (\text{{epsilon\_start}} - \text{{epsilon\_end}}) \times \text{{decay\_rate}} + \text{{epsilon\_end}} \]

This is a simple linear interpolation.

## E-greedy
The e-greedy algorithm breaks ties when more than one state-actions pairs have the same value, using a random function.

## Training
The model can be read from file and store to a file every time.

## Testing
You can specify the total amount of episodes, also every how many episodes you want to evaluate the model, and on each evaluation, for how many episodes you want to test. A small diagram showing this is

![](./docs/training-testing-diagram.png)

## Storing actions
You can use the parameter `--store_actions` to store the actions, state, reward, end and info to a file for later checking. Be careful that if you do this during training or a long testing it can get very large.

## MlFlow
It uses mlflow to log the experiments and you can say which Experiment ID it is. Automatically will store the commit hash of the NetSecEnvAgents repository and the NetSecEnv repository.



