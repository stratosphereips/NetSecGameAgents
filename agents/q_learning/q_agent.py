# Authors:  Ondrej Lukas - ondrej.lukas@aic.fel.cvut.cz
#           Arti
#           Sebastian Garcia. sebastian.garcia@agents.fel.cvut.cz
import sys
import os
import numpy as np
import random
import pickle
import argparse
import logging
# This is used so the agent can see the environment and game component
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ) ) )))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) )))

# This is used so the agent can see the environment and game component
# with the path fixed, we can import now
from env.game_components import Action, Observation, GameState
from base_agent import BaseAgent
from agent_utils import generate_valid_actions, state_as_ordered_string


class QAgent(BaseAgent):

    def __init__(self, host, port, role="Attacker", alpha=0.1, gamma=0.6, epsilon=0.1) -> None:
        super().__init__(host, port, role)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values = {}
        self._str_to_id = {}

    def store_q_table(self,filename):
        with open(filename, "wb") as f:
            data = {"q_table":self.q_values, "state_mapping": self._str_to_id}
            pickle.dump(data, f)

    def load_q_table(self,filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.q_values = data["q_table"]
            self._str_to_id = data["state_mapping"]

    def get_state_id(self, state:GameState) -> int:
        state_str = state_as_ordered_string(state)
        if state_str not in self._str_to_id:
            self._str_to_id[state_str] = len(self._str_to_id)
        return self._str_to_id[state_str]
    
    def max_action_q(self, observation:Observation) -> Action:
        state = observation.state
        actions = generate_valid_actions(state)
        state_id = self.get_state_id(state)
        tmp = dict(((state_id, a), self.q_values.get((state_id, a), 0)) for a in actions)
        return tmp[max(tmp,key=tmp.get)] #return maximum Q_value for a given state (out of available actions)
   
    def select_action(self, observation:Observation, testing=False) -> Action:
        state = observation.state
        actions = generate_valid_actions(state)
        state_id = self.get_state_id(state)
        
        # E-greedy play. If the random number is less than the e, then choose random to explore.
        # But do not do it if we are testing a model. 
        if random.uniform(0, 1) <= self.epsilon and not testing:
            # We are training
            # Random choose an ation from the list of actions?
            action = random.choice(list(actions))
            if (state_id, action) not in self.q_values:
                self.q_values[state_id, action] = 0
            return action, state_id
        else: 
            # We are training
            # Select the action with highest q_value
            # The default initial q-value for a (state, action) pair is 0.
            initial_q_value = 0
            tmp = dict(((state_id, action), self.q_values.get((state_id, action), initial_q_value)) for action in actions)
            ((state_id, action), value) = max(tmp.items(), key=lambda x: (x[1], random.random()))
            #if max_q_key not in self.q_values:
            try:
                self.q_values[state_id, action]
            except KeyError:
                self.q_values[state_id, action] = 0
            return action, state_id
        
    def play_game(self, observation, num_episodes=1, testing=False):
        """
        The main function for the gameplay. Handles the main interaction loop.
        """
        returns = []
        num_steps = 0
        for episode in range(num_episodes):
            episodic_rewards = []
            while observation and not observation.end:
                self._logger.debug(f'Observation received:{observation}')
                # Store steps so far
                num_steps += 1
                # Get next_action. If we are not training, selection is different, so pass it
                action, state_id = self.select_action(observation, testing)
                # Perform the action and observe next observation
                observation = self.make_step(action)
                # Store the reward of the next observation
                episodic_rewards.append(observation.reward)
                if not testing:
                    # If we are training update the Q-table
                    self.q_values[state_id, action] += self.alpha * (observation.reward + self.gamma * self.max_action_q(observation)) - self.q_values[state_id, action]
                # Copy the last observation so we can return it and avoid the empty observation after the reset
                last_observation = observation
            # Sum all episodic returns 
            returns.append(np.sum(episodic_rewards))
            self._logger.info(f"Episode {episode} (len={len(episodic_rewards)}) ended with sum of rewards {np.sum(episodic_rewards)}. For all past episodes: mean returns = {np.mean(returns):.5f} ± {np.std(returns):.5f} Nr. States in Q_table = {len(self.q_values)}")
            # Reset the episode
            observation = self.request_game_reset()
        agent._logger.info(f"Final results for {self.__class__.__name__} after {num_episodes} episodes: {np.mean(returns)}±{np.std(returns)}")
        # This will be the last observation played before the reset
        return (last_observation, num_steps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('You can train the agent, or test it. \n Test is also to use the agent. \n During training and testing the performance is logged.')
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    parser.add_argument("--episodes", help="Sets number of episodes to run.", default=1000, type=int)
    parser.add_argument("--test_each", help="Evaluate the performance every this number of episodes. During training and testing.", default=100, type=int)
    parser.add_argument("--epsilon", help="Sets epsilon for exploration during training.", default=0.2, type=float)
    parser.add_argument("--gamma", help="Sets gamma discount for Q-learing during training.", default=0.9, type=float)
    parser.add_argument("--alpha", help="Sets alpha for learning rate during training.", default=0.1, type=float)
    parser.add_argument("--logdir", help="Folder to store logs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    parser.add_argument("--previous_model", help="Load the previous model. If training, it will start from here. If testing, will use to test.", default='./q_agent_marl.pickle', type=str)
    parser.add_argument("--testing", help="Test the agent. No train.", default=False)
    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logging.basicConfig(filename=os.path.join(args.logdir, "q_agent.log"), filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',level=logging.INFO)

    # Create agent
    agent = QAgent(args.host, args.port, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)

    # If there is a previous model passed. Always use it for both training and testing.
    if args.previous_model:
        # Load table
        agent._logger.info(f'Loading the previous model in file {args.previous_model}')
        agent.load_q_table(args.previous_model)

    if not args.testing:
        # Training
        # Register
        observation = agent.register()
        # Play
        agent.play_game(observation, args.episodes, testing=args.testing)       
        # Store the q-table
        agent.store_q_table(args.previous_model)
        # Terminate
        agent._logger.info("Terminating interaction")
        agent.terminate_connection()

    elif not args.testing:
        agent.play_game(args.episodes, testing=args.testing)
        agent.store_q_table("./q_agent_marl.pickle")

#     logger.info(f'Initializing the environment')
#     env = NetworkSecurityEnvironment(args.task_config_file)

#     observation = env.reset()

#     logger.info('Creating the agent')
#     agent = QAgent(env, args.alpha, args.gamma, args.epsilon)
#     if args.filename:
#         try:
#             # Load a previous qtable from a pickled file
#             logger.info(f'Loading a previous Qtable')
#             agent.load_q_table(args.filename)
#         except FileNotFoundError:
#             logger.info(f"No previous qtable file found to load, starting with an emptly zeroed qtable")

#     # If we are not evaluating the model
#     if not args.test:
#         # Run for some episodes
#         logger.info(f'Starting the training')
#         for i in range(1, args.episodes + 1):
#             # Reset
#             observation = env.reset()
#             # Play complete round
#             ret, win,_,_ = agent.play(observation)
#             logger.info(f'Reward: {ret}, Win:{win}')
#             # Every X episodes, eval
#             if i % args.eval_each == 0:
#                 wins = 0
#                 detected = 0
#                 returns = []
#                 num_steps = []
#                 num_win_steps = []
#                 num_detected_steps = []
#                 for j in range(args.eval_for):
#                     observation = env.reset()
#                     ret, win, detection, steps = agent.evaluate(observation)
#                     if win:
#                         wins += 1
#                         num_win_steps += [steps]
#                     if detection:
#                         detected += 1
#                         num_detected_steps += [steps]
#                     returns += [ret]
#                     num_steps += [steps]

#                 eval_win_rate = (wins/(args.eval_for+1))*100
#                 eval_detection_rate = (detected/(args.eval_for+1))*100
#                 eval_average_returns = np.mean(returns)
#                 eval_std_returns = np.std(returns)
#                 eval_average_episode_steps = np.mean(num_steps)
#                 eval_std_episode_steps = np.std(num_steps)
#                 eval_average_win_steps = np.mean(num_win_steps)
#                 eval_std_win_steps = np.std(num_win_steps)
#                 eval_average_detected_steps = np.mean(num_detected_steps)
#                 eval_std_detected_steps = np.std(num_detected_steps)

#                 text = f'''Evaluated after {i} episodes, for {args.eval_for} episodes.
#                     Wins={wins},
#                     Detections={detected},
#                     winrate={eval_win_rate:.3f}%,
#                     detection_rate={eval_detection_rate:.3f}%,
#                     average_returns={eval_average_returns:.3f} +- {eval_std_returns:.3f},
#                     average_episode_steps={eval_average_episode_steps:.3f} +- {eval_std_episode_steps:.3f},
#                     average_win_steps={eval_average_win_steps:.3f} +- {eval_std_win_steps:.3f},
#                     average_detected_steps={eval_average_detected_steps:.3f} +- {eval_std_detected_steps:.3f}
#                     '''
#                 print(text)
#                 logger.info(text)
#                 # Store in tensorboard
#                 writer.add_scalar("charts/eval_avg_win_rate", eval_win_rate, i)
#                 writer.add_scalar("charts/eval_avg_detection_rate", eval_detection_rate, i)
#                 writer.add_scalar("charts/eval_avg_returns", eval_average_returns , i)
#                 writer.add_scalar("charts/eval_std_returns", eval_std_returns , i)
#                 writer.add_scalar("charts/eval_avg_episode_steps", eval_average_episode_steps , i)
#                 writer.add_scalar("charts/eval_std_episode_steps", eval_std_episode_steps , i)
#                 writer.add_scalar("charts/eval_avg_win_steps", eval_average_win_steps , i)
#                 writer.add_scalar("charts/eval_std_win_steps", eval_std_win_steps , i)
#                 writer.add_scalar("charts/eval_avg_detected_steps", eval_average_detected_steps , i)
#                 writer.add_scalar("charts/eval_std_detected_steps", eval_std_detected_steps , i)

#         # Store the q table on disk
#         if args.filename:
#             agent.store_q_table(args.filename)
#     if args.filename:
#         agent = QAgent(env, args.alpha, args.gamma, args.epsilon)
#         agent.load_q_table(args.filename)

#     # Test
#     wins = 0
#     detected = 0
#     returns = []
#     num_steps = []
#     num_win_steps = []
#     num_detected_steps = []
#     for i in range(args.test_for + 1):
#         observation = env.reset()
#         ret, win, detection, steps = agent.evaluate(observation)
#         if win:
#             wins += 1
#             num_win_steps += [steps]
#         if detection:
#             detected +=1
#             num_detected_steps += [steps]
#         returns += [ret]
#         num_steps += [steps]

#         test_win_rate = (wins/(i+1))*100
#         test_detection_rate = (detected/(i+1))*100
#         test_average_returns = np.mean(returns)
#         test_std_returns = np.std(returns)
#         test_average_episode_steps = np.mean(num_steps)
#         test_std_episode_steps = np.std(num_steps)
#         test_average_win_steps = np.mean(num_win_steps)
#         test_std_win_steps = np.std(num_win_steps)
#         test_average_detected_steps = np.mean(num_detected_steps)
#         test_std_detected_steps = np.std(num_detected_steps)


#         # Print and report every 100 test episodes
#         if i % 100 == 0 and i != 0:
#             text = f'''Test results after {i} episodes.
#                 Wins={wins},
#                 Detections={detected},
#                 winrate={test_win_rate:.3f}%,
#                 detection_rate={test_detection_rate:.3f}%,
#                 average_returns={test_average_returns:.3f} +- {test_std_returns:.3f},
#                 average_episode_steps={test_average_episode_steps:.3f} +- {test_std_episode_steps:.3f},
#                 average_win_steps={test_average_win_steps:.3f} +- {test_std_win_steps:.3f},
#                 average_detected_steps={test_average_detected_steps:.3f} +- {test_std_detected_steps:.3f}
#                 '''

#             print(text)
#             logger.info(text)

#         # Store in tensorboard
#         writer.add_scalar("charts/test_avg_win_rate", test_win_rate, i)
#         writer.add_scalar("charts/test_avg_detection_rate", test_detection_rate, i)
#         writer.add_scalar("charts/test_avg_returns", test_average_returns , i)
#         writer.add_scalar("charts/test_std_returns", test_std_returns , i)
#         writer.add_scalar("charts/test_avg_episode_steps", test_average_episode_steps , i)
#         writer.add_scalar("charts/test_std_episode_steps", test_std_episode_steps , i)
#         writer.add_scalar("charts/test_avg_win_steps", test_average_win_steps , i)
#         writer.add_scalar("charts/test_std_win_steps", test_std_win_steps , i)
#         writer.add_scalar("charts/test_avg_detected_steps", test_average_detected_steps , i)
#         writer.add_scalar("charts/test_std_detected_steps", test_std_detected_steps , i)


#     text = f'''Final test after {i} episodes
#         Wins={wins},
#         Detections={detected},
#         winrate={test_win_rate:.3f}%,
#         detection_rate={test_detection_rate:.3f}%,
#         average_returns={test_average_returns:.3f} +- {test_std_returns:.3f},
#         average_episode_steps={test_average_episode_steps:.3f} +- {test_std_episode_steps:.3f},
#         average_win_steps={test_average_win_steps:.3f} +- {test_std_win_steps:.3f},
#         average_detected_steps={test_average_detected_steps:.3f} +- {test_std_detected_steps:.3f}
#         '''
#     print(text)
#     logger.info(text)