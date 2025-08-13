# Authors:  Ondrej Lukas - ondrej.lukas@aic.fel.cvut.cz
#           Arti
#           Sebastian Garcia. sebastian.garcia@agents.fel.cvut.cz
import sys
import numpy as np
import random
import pickle
import argparse
import logging
import wandb
import subprocess
import time

from os import path, makedirs
# with the path fixed, we can import now
from AIDojoCoordinator.game_components import Action, Observation, GameState, AgentStatus
from NetSecGameAgents.agents.base_agent import BaseAgent
from NetSecGameAgents.agents.agent_utils import generate_valid_actions, state_as_ordered_string

class QAgent(BaseAgent):

    def __init__(self, host, port, role="Attacker", alpha=0.1, gamma=0.6, epsilon_start=0.9, epsilon_end=0.1, epsilon_max_episodes=5000, apm_limit:int=None) -> None:
        super().__init__(host, port, role)
        self.alpha = alpha
        self.gamma = gamma
        self.q_values = {}
        self._str_to_id = {}
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_max_episodes = epsilon_max_episodes
        self.current_epsilon = epsilon_start
        self._apm_limit = apm_limit
        if self._apm_limit:
            self.inter_action_interval = 60/apm_limit
        else:
            self.inter_action_interval = 0

    def store_q_table(self, filename):
        with open(filename, "wb") as f:
            data = {"q_table":self.q_values, "state_mapping": self._str_to_id}
            pickle.dump(data, f)

    def load_q_table(self,filename):
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.q_values = data["q_table"]
                self._str_to_id = data["state_mapping"]
            self._logger.info(f'Successfully loading file {filename}')
        except Exception as e:
            self._logger.info(f'Error loading file {filename}. {e}')
            sys.exit(-1)

    def get_state_id(self, state:GameState) -> int:
        # Here the state has to be ordered, so different orders are not taken as two different states.
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
   
    def select_action(self, observation:Observation, testing=False) -> tuple:
        state = observation.state
        actions = generate_valid_actions(state)
        state_id = self.get_state_id(state)
        
        # E-greedy play. If the random number is less than the e, then choose random to explore.
        # But do not do it if we are testing a model. 
        if random.uniform(0, 1) <= self.current_epsilon and not testing:
            # We are training
            # Random choose an ation from the list of actions?
            action = random.choice(list(actions))
            if (state_id, action) not in self.q_values:
                self.q_values[state_id, action] = 0
            return action, state_id
        else: 
            # Here we can be during training outside the e-greede, or during testing
            # Select the action with highest q_value, or random pick to break the ties
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

    def recompute_reward(self, observation: Observation) -> Observation:
        """
        Redefine how q-learning recomputes the inner reward
        """
        new_observation = None
        state = observation.state
        reward = observation.reward
        end = observation.end
        info = observation.info

        if info and info['end_reason'] == AgentStatus.Fail:
            reward = -1000
        elif info and info['end_reason'] == AgentStatus.Success:
            reward = 1000
        elif info and info['end_reason'] == AgentStatus.TimeoutReached:
            reward = -100
        else:
            reward = -1
        
        new_observation = Observation(state, reward, end, info)
        return new_observation

    def update_epsilon_with_decay(self, episode_number)->float:
        decay_rate = np.max([(self.epsilon_max_episodes - episode_number) / self.epsilon_max_episodes, 0])
        new_eps = (self.epsilon_start - self.epsilon_end ) * decay_rate + self.epsilon_end
        self.logger.debug(f"Updating epsilon - new value:{new_eps}")
        return new_eps
    
    def play_game(self, observation, episode_num, testing=False):
        """
        The main function for the gameplay. Handles the main interaction loop.
        """
        num_steps = 0
        # Run the whole episode
        while not observation.end:
            # Store steps so far
            num_steps += 1
            start_time = time.time()
            # Get next action. If we are not training, selection is different, so pass it as argument
            action, state_id = self.select_action(observation, testing)
            if args.store_actions:
                actions_logger.info(f"\tState:{observation.state}")
                actions_logger.info(f"\tEnd:{observation.end}")
                actions_logger.info(f"\tInfo:{observation.info}")
            self.logger.info(f"Action selected:{action}")
            # Perform the action and observe next observation
            observation = self.make_step(action)
           
            # Recompute the rewards
            observation = self.recompute_reward(observation)
            if not testing:
                # If we are training update the Q-table
                self.q_values[state_id, action] += self.alpha * (observation.reward + self.gamma * self.max_action_q(observation)) - self.q_values[state_id, action]

            # Check the apm (actions per minute)
            if self._apm_limit:
                elapsed_time = time.time() - start_time
                remaining_time = self.inter_action_interval - elapsed_time
                if remaining_time > 0:
                    # We still have some time in this interval, but we can not
                    # take more actions. So wait until the next interval starts
                    self._logger.debug(f"Waiting for {remaining_time}s before next action.")
                    time.sleep(remaining_time)
                start_time = time.time()

        if args.store_actions:
            actions_logger.info(f"\t State:{observation.state}")
            actions_logger.info(f"\t End:{observation.end}")
            actions_logger.info(f"\t Info:{observation.info}")
        # update epsilon value
        if not testing:
            self.current_epsilon = self.update_epsilon_with_decay(episode_num)
        # Reset the episode
        _ = self.request_game_reset()
        # This will be the last observation played before the reset
        return observation, num_steps

if __name__ == '__main__':
    parser = argparse.ArgumentParser('You can train the agent, or test it. \n Test is also to use the agent. \n During training and testing the performance is logged.')
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    parser.add_argument("--episodes", help="Sets number of episodes to run.", default=15000, type=int)
    parser.add_argument("--test_each", help="Evaluate the performance every this number of episodes. During training and testing.", default=1000, type=int)
    parser.add_argument("--test_for", help="Evaluate the performance for this number of episodes each time. Only during training.", default=250, type=int)
    parser.add_argument("--epsilon_start", help="Sets the start epsilon for exploration during training.", default=0.9, type=float)
    parser.add_argument("--epsilon_end", help="Sets the end epsilon for exploration during training.", default=0.1, type=float)
    parser.add_argument("--epsilon_max_episodes", help="Max episodes for epsilon to reach maximum decay", default=8000, type=int)
    parser.add_argument("--gamma", help="Sets gamma discount for Q-learing during training.", default=0.9, type=float)
    parser.add_argument("--alpha", help="Sets alpha for learning rate during training.", default=0.1, type=float)
    parser.add_argument("--logdir", help="Folder to store logs", default=path.join(path.dirname(path.abspath(__file__)), "logs"))
    parser.add_argument("--previous_model", help="Load the previous model. If training, it will start from here. If testing, will use to test.", type=str)
    parser.add_argument("--testing", help="Test the agent. No train.", default=False, type=bool)
    parser.add_argument("--experiment_id", help="Id of the experiment to record into Weights & Biases.", default='', type=str)
    parser.add_argument("--store_actions", help="Store actions in the log file q_agents_actions.log.", default=False, type=bool)
    parser.add_argument("--store_models_every", help="Store a model to disk every these number of episodes.", default=2000, type=int)
    parser.add_argument("--env_conf", help="Configuration file of the env. Only for logging purposes.", required=False, default='./env/netsecenv_conf.yaml', type=str)
    parser.add_argument("--early_stop_threshold", help="Threshold for win rate for testing. If the value goes over this threshold, the training is stopped. Defaults to 95 (mean 95%% perc)", required=False, default=95, type=float)
    parser.add_argument("--apm", help="Actions per minute", default=10000, type=int, required=False)
    args = parser.parse_args()

    if not path.exists(args.logdir):
        makedirs(args.logdir)
    logging.basicConfig(filename=path.join(args.logdir, "q_agent.log"), filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',level=logging.INFO)

    # Create agent
    agent = QAgent(args.host, args.port, alpha=args.alpha, gamma=args.gamma, epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end, epsilon_max_episodes=args.epsilon_max_episodes, apm_limit=args.apm)

    # Log for Actions. After agent creation
    actions_logger = logging.getLogger('QAgentActions')
    actions_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    actions_handler = logging.FileHandler(path.join(args.logdir, "q_agent_actions.log"), mode="w")
    actions_handler.setLevel(logging.INFO)  
    actions_handler.setFormatter(formatter)
    actions_logger.addHandler(actions_handler)

    # Early stop flag
    early_stop = False

    # If there is a previous model passed. Always use it for both training and testing.
    if args.previous_model:
        # Load table
        agent._logger.info(f'Loading the previous model in file {args.previous_model}')
        try:
            agent.load_q_table(args.previous_model)
        except FileNotFoundError:
            message = f'Problem loading the file: {args.previous_model}'
            agent._logger.info(message)
            print(message)


    if not args.testing:
        # Wandb experiment name
        experiment_name = "Training and Eval of Q-learning Agent"
    elif args.testing:
        # Evaluate the agent performance

        # Wandb experiment name
        experiment_name = "Testing of Q-learning Agent"


    # This code runs for both training and testing. The difference is in the args.testing variable that is passed along
    # How it works:
    # - Evaluate for several 'episodes' (parameter)
    # - Each episode finishes with: steps played, return, win/lose. Store all
    # - Each episode compute the avg and std of all.
    # - Every X episodes (parameter), report in log and wandb
    # - At the end, report in log and wandb and console

    # Register the agent
    observation = agent.register()

    try:
        # Initialize wandb
        wandb.init(
            entity='Stratosphere',
            project='UTEP-Collaboration',
            group='sebas-qlearning',
            name=experiment_name + f'. ID {args.experiment_id}'
        )

        # To keep statistics of each episode
        wins = 0
        detected = 0
        max_steps = 0
        num_win_steps = []
        num_detected_steps = []
        num_max_steps_steps = []
        num_detected_returns = []
        num_win_returns = []
        num_max_steps_returns = []

        # Configure wandb with parameters and tags
        wandb.config.update({
            "alpha": args.alpha,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_max_episodes": args.epsilon_max_episodes,
            "gamma": args.gamma,
            "episodes": args.episodes,
            "test_each": args.test_each,
            "test_for": args.test_for,
            "testing": args.testing,
            "experiment_name": experiment_name,
            "notes": "This is an evaluation"
        })

        if args.previous_model:
            wandb.config.update({"previous_model_loaded": str(args.previous_model)})

        # Use subprocess.run to get the commit hash
        netsecenv_command = "git rev-parse HEAD"
        netsecenv_git_result = subprocess.run(netsecenv_command, shell=True, capture_output=True, text=True).stdout
        agents_command = "cd NetSecGameAgents; git rev-parse HEAD"
        agents_git_result = subprocess.run(agents_command, shell=True, capture_output=True, text=True).stdout
        agent._logger.info(f'Using commits. NetSecEnv: {netsecenv_git_result}. Agents: {agents_git_result}')
        wandb.config.update({
            "netsecenv_commit": netsecenv_git_result.strip(),
            "agents_commit": agents_git_result.strip()
        })
        # Log the env conf
        try:
            if path.exists(args.env_conf):
                wandb.save(args.env_conf, base_path=path.dirname(path.abspath(args.env_conf)))
            else:
                agent._logger.warning(f"Environment config file not found: {args.env_conf}")
                wandb.config.update({"env_conf_path": args.env_conf})
        except Exception as e:
            agent._logger.warning(f"Could not save env config file: {e}")
            wandb.config.update({"env_conf_path": args.env_conf})
        agent._logger.info(f'Epsilon Start: {agent.epsilon_start}')
        agent._logger.info(f'Epsilon End: {agent.epsilon_end}')
        agent._logger.info(f'Epsilon Max Episodes: {agent.epsilon_max_episodes}')

        for episode in range(1, args.episodes + 1):
                if not early_stop:
                    # Play 1 episode
                    observation, num_steps = agent.play_game(observation, testing=args.testing, episode_num=episode)       

                    state = observation.state
                    reward = observation.reward
                    end = observation.end
                    info = observation.info

                    if observation.info and observation.info['end_reason'] == AgentStatus.Fail:
                        detected +=1
                        num_detected_steps += [num_steps]
                        num_detected_returns += [reward]
                    elif observation.info and observation.info['end_reason'] == AgentStatus.Success:
                        wins += 1
                        num_win_steps += [num_steps]
                        num_win_returns += [reward]
                    elif observation.info and observation.info['end_reason'] == AgentStatus.TimeoutReached:
                        max_steps += 1
                        num_max_steps_steps += [num_steps]
                        num_max_steps_returns += [reward]

                    if args.testing:
                        agent._logger.error(f"Testing episode {episode}: Steps={num_steps}. Reward {reward}. States in Q_table = {len(agent.q_values)}")
                    elif not args.testing:
                        agent._logger.error(f"Training episode {episode}: Steps={num_steps}. Reward {reward}. States in Q_table = {len(agent.q_values)}")

                    # Reset the game
                    observation = agent.request_game_reset()

                    eval_win_rate = (wins/episode) * 100
                    eval_detection_rate = (detected/episode) * 100
                    eval_average_returns = np.mean(num_detected_returns + num_win_returns + num_max_steps_returns)
                    eval_std_returns = np.std(num_detected_returns + num_win_returns + num_max_steps_returns)
                    eval_average_episode_steps = np.mean(num_win_steps + num_detected_steps + num_max_steps_steps)
                    eval_std_episode_steps = np.std(num_win_steps + num_detected_steps + num_max_steps_steps)
                    eval_average_win_steps = np.mean(num_win_steps)
                    eval_std_win_steps = np.std(num_win_steps)
                    eval_average_detected_steps = np.mean(num_detected_steps)
                    eval_std_detected_steps = np.std(num_detected_steps)
                    eval_average_max_steps_steps = np.mean(num_max_steps_steps)
                    eval_std_max_steps_steps = np.std(num_max_steps_steps)

                    # Log results for testing mode every episode
                    if args.testing:
                        wandb.log({
                            "test_avg_win_rate": eval_win_rate,
                            "test_avg_detection_rate": eval_detection_rate,
                            "test_avg_returns": eval_average_returns,
                            "test_std_returns": eval_std_returns,
                            "test_avg_episode_steps": eval_average_episode_steps,
                            "test_std_episode_steps": eval_std_episode_steps,
                            "test_avg_win_steps": eval_average_win_steps,
                            "test_std_win_steps": eval_std_win_steps,
                            "test_avg_detected_steps": eval_average_detected_steps,
                            "test_std_detected_steps": eval_std_detected_steps,
                            "test_avg_max_steps_steps": eval_average_max_steps_steps,
                            "test_std_max_steps_steps": eval_std_max_steps_steps,
                            "current_episode": episode
                        }, step=episode)

                    # Now Test, log and report. This happens every X training episodes
                    if episode % args.test_each == 0 and episode != 0:
                        # If we are training, every these number of episodes, we need to test for some episodes.
                        # If we are testing, it is not necessary since the model does not change
                        if not args.testing:
                            # This test happens during a training

                            # First report performance of trained model up to here
                            text = f'''Performance evaluated after {episode} training episodes.
                                Wins={wins},
                                Detections={detected},
                                winrate={eval_win_rate:.3f}%,
                                detection_rate={eval_detection_rate:.3f}%,
                                average_returns={eval_average_returns:.3f} +- {eval_std_returns:.3f},
                                average_episode_steps={eval_average_episode_steps:.3f} +- {eval_std_episode_steps:.3f},
                                average_win_steps={eval_average_win_steps:.3f} +- {eval_std_win_steps:.3f},
                                average_detected_steps={eval_average_detected_steps:.3f} +- {eval_std_detected_steps:.3f}
                                average_max_steps_steps={eval_std_max_steps_steps:.3f} +- {eval_std_max_steps_steps:.3f},
                                epsilon={agent.current_epsilon}
                                '''
                            agent._logger.info(text)
                            wandb.log({
                                "eval_avg_win_rate": eval_win_rate,
                                "eval_avg_detection_rate": eval_detection_rate,
                                "eval_avg_returns": eval_average_returns,
                                "eval_std_returns": eval_std_returns,
                                "eval_avg_episode_steps": eval_average_episode_steps,
                                "eval_std_episode_steps": eval_std_episode_steps,
                                "eval_avg_win_steps": eval_average_win_steps,
                                "eval_std_win_steps": eval_std_win_steps,
                                "eval_avg_detected_steps": eval_average_detected_steps,
                                "eval_std_detected_steps": eval_std_detected_steps,
                                "eval_avg_max_steps_steps": eval_average_max_steps_steps,
                                "eval_std_max_steps_steps": eval_std_max_steps_steps,
                                "current_epsilon": agent.current_epsilon,
                                "current_episode": episode
                            }, step=episode)

                            # To keep statistics of testing each episode
                            test_wins = 0
                            test_detected = 0
                            test_max_steps = 0
                            test_num_win_steps = []
                            test_num_detected_steps = []
                            test_num_max_steps_steps = []
                            test_num_detected_returns = []
                            test_num_win_returns = []
                            test_num_max_steps_returns = []

                            # Test
                            for test_episode in range(1, args.test_for + 1):
                                # Play 1 episode
                                # See that we force the model to freeze by telling it that it is in 'testing' mode.
                                # Also the episode_num is not updated since this controls the decay of the epsilon during training and we dont want to change that
                                test_observation, test_num_steps = agent.play_game(observation, testing=True, episode_num=episode)       

                                test_state = test_observation.state
                                test_reward = test_observation.reward
                                test_end = test_observation.end
                                test_info = test_observation.info

                                if test_info and test_info['end_reason'] == AgentStatus.Fail:
                                    test_detected +=1
                                    test_num_detected_steps += [num_steps]
                                    test_num_detected_returns += [reward]
                                elif test_info and test_info['end_reason'] == AgentStatus.Success:
                                    test_wins += 1
                                    test_num_win_steps += [num_steps]
                                    test_num_win_returns += [reward]
                                elif test_info and test_info['end_reason'] == AgentStatus.TimeoutReached:
                                    test_max_steps += 1
                                    test_num_max_steps_steps += [num_steps]
                                    test_num_max_steps_returns += [reward]

                                agent._logger.error(f"\tTesting episode {test_episode}: Steps={test_num_steps}. Reward {test_reward}. States in Q_table = {len(agent.q_values)}")

                                # Reset the game
                                test_observation = agent.request_game_reset()

                                test_win_rate = (test_wins/test_episode) * 100
                                test_detection_rate = (test_detected/test_episode) * 100
                                test_average_returns = np.mean(test_num_detected_returns + test_num_win_returns + test_num_max_steps_returns)
                                test_std_returns = np.std(test_num_detected_returns + test_num_win_returns + test_num_max_steps_returns)
                                test_average_episode_steps = np.mean(test_num_win_steps + test_num_detected_steps + test_num_max_steps_steps)
                                test_std_episode_steps = np.std(test_num_win_steps + test_num_detected_steps + test_num_max_steps_steps)
                                test_average_win_steps = np.mean(test_num_win_steps)
                                test_std_win_steps = np.std(test_num_win_steps)
                                test_average_detected_steps = np.mean(test_num_detected_steps)
                                test_std_detected_steps = np.std(test_num_detected_steps)
                                test_average_max_steps_steps = np.mean(test_num_max_steps_steps)
                                test_std_max_steps_steps = np.std(test_num_max_steps_steps)

                                # store model. Use episode (training counter) and not test_episode (test counter)
                                if episode % args.store_models_every == 0 and episode != 0:
                                    agent.store_q_table(f'/data/AIDojo/Models/q_agent_marl.experiment{args.experiment_id}-episodes-{episode}.pickle')

                            text = f'''Tested for {test_episode} episodes after {episode} training episode.
                                Wins={test_wins},
                                Detections={test_detected},
                                winrate={test_win_rate:.3f}%,
                                detection_rate={test_detection_rate:.3f}%,
                                average_returns={test_average_returns:.3f} +- {test_std_returns:.3f},
                                average_episode_steps={test_average_episode_steps:.3f} +- {test_std_episode_steps:.3f},
                                average_win_steps={test_average_win_steps:.3f} +- {test_std_win_steps:.3f},
                                average_detected_steps={test_average_detected_steps:.3f} +- {test_std_detected_steps:.3f}
                                average_max_steps_steps={test_std_max_steps_steps:.3f} +- {test_std_max_steps_steps:.3f},
                                epsilon={agent.current_epsilon}
                                '''
                            agent._logger.info(text)
                            print(text)
                            # Store in wandb
                            wandb.log({
                                "test_avg_win_rate": test_win_rate,
                                "test_avg_detection_rate": test_detection_rate,
                                "test_avg_returns": test_average_returns,
                                "test_std_returns": test_std_returns,
                                "test_avg_episode_steps": test_average_episode_steps,
                                "test_std_episode_steps": test_std_episode_steps,
                                "test_avg_win_steps": test_average_win_steps,
                                "test_std_win_steps": test_std_win_steps,
                                "test_avg_detected_steps": test_average_detected_steps,
                                "test_std_detected_steps": test_std_detected_steps,
                                "test_avg_max_steps_steps": test_average_max_steps_steps,
                                "test_std_max_steps_steps": test_std_max_steps_steps,
                                "test_current_epsilon": agent.current_epsilon,
                                "test_current_episode": episode
                            }, step=episode)

                            if test_win_rate >= args.early_stop_threshold:
                                agent.logger.info(f'Early stopping. Test win rate: {test_win_rate}. Threshold: {args.early_stop_threshold}')
                                early_stop = True


        # Log the last final episode when it ends
        text = f'''Final model performance after {episode} episodes.
            Wins={wins},
            Detections={detected},
            winrate={eval_win_rate:.3f}%,
            detection_rate={eval_detection_rate:.3f}%,
            average_returns={eval_average_returns:.3f} +- {eval_std_returns:.3f},
            average_episode_steps={eval_average_episode_steps:.3f} +- {eval_std_episode_steps:.3f},
            average_win_steps={eval_average_win_steps:.3f} +- {eval_std_win_steps:.3f},
            average_detected_steps={eval_average_detected_steps:.3f} +- {eval_std_detected_steps:.3f}
            average_max_steps_steps={eval_std_max_steps_steps:.3f} +- {eval_std_max_steps_steps:.3f},
            epsilon={agent.current_epsilon}
            '''

        agent._logger.info(text)
        print(text)
        agent._logger.error("Terminating interaction")
        agent.terminate_connection()

        # Finish wandb run
        wandb.finish()

    except KeyboardInterrupt:
        # Store the q-table
        # Just in case...
        if not args.testing:
            agent.store_q_table(f'q_agent_marl.experiment{args.experiment_id}.pickle')
    finally:
        # Store the q-table
        if not args.testing:
            agent.store_q_table(f'q_agent_marl.experiment{args.experiment_id}.pickle')