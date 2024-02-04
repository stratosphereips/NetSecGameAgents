#Author: Ondrej Lukas, ondrej.lukas@aic.cvut.cz
# This agents just randomnly picks actions. No learning
import sys
import logging
import os
from random import choice
import argparse
import numpy as np
import mlflow

# This is used so the agent can see the environment and game components
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ) ) )))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) )))
# with the path fixed, we can import now
from env.game_components import Action, Observation, GameState
from base_agent import BaseAgent
from agent_utils import generate_valid_actions
from datetime import datetime


class RandomAgent(BaseAgent):

    def __init__(self, host, port,role, seed) -> None:
        super().__init__(host, port, role)
    

    def play_game(self, num_episodes=1):
        """
        The main function for the gameplay. Handles agent registration and the main interaction loop.
        """
        
        observation = self.register()
        returns = []
        for episode in range(num_episodes):
            self._logger.info(f"Playing episode {episode}")
            episodic_returns = []
            while observation and not observation.end:
                self._logger.debug(f'Observation received:{observation}')
                # Store returns in the episode
                episodic_returns.append(observation.reward)
                # Select the action randomly
                action = self.select_action(observation)
                observation = self.make_step(action)
            self._logger.debug(f'Observation received:{observation}')
            returns.append(np.sum(episodic_returns))
            self._logger.info(f"Episode {episode} ended with return{np.sum(episodic_returns)}. Mean returns={np.mean(returns)}±{np.std(returns)}")
            # Reset the episode
            observation = self.request_game_reset()
        self._logger.info(f"Final results for {self.__class__.__name__} after {num_episodes} episodes: {np.mean(returns)}±{np.std(returns)}")
        self._logger.info("Terminating interaction")
        self.terminate_connection()
    
    def select_action(self, observation:Observation)->Action:
        valid_actions = generate_valid_actions(observation.state)
        action = choice(valid_actions)
        return action

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    parser.add_argument("--episodes", help="Sets number of testing episodes", default=10, type=int)
    parser.add_argument("--test_each", help="Sets periodic evaluation during testing", default=100, type=int)
    parser.add_argument("--num_trials", type=int, default=1, help="Number of experiments to run")
    parser.add_argument("--logdir", help="Folder to store logs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    parser.add_argument("--evaluate", help="Evaluate the agent", default=True)
    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logging.basicConfig(filename=os.path.join(args.logdir, "random_agent.log"), filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',level=logging.INFO)

    # Create agent
    agent = RandomAgent(args.host, args.port,"Attacker", seed=42)

    if not args.evaluate:
        agent.play_game(args.episodes)
    else:
        # Evaluate the agent
        # Evaluate for several runs
        # Each run is composed of many episodes
        # Every X episodes, report in log and console

        experiment_name = "Evaluation of Random Agent"
        mlflow.set_experiment(experiment_name)

        # To keep statistics of all trials
        trial_win_rate = []
        trial_detection_rate = []
        trial_average_returns = []
        trial_std_returns = []
        trial_average_episode_steps = []
        trial_std_episode_steps = []
        trial_average_win_steps = []
        trial_std_win_steps = []
        trial_average_detected_steps = []
        trial_std_detected_steps = []

        for run_number in range(args.num_trials):
            run_name = f"Run_{run_number}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            with mlflow.start_run(run_name=run_name) as run:
                agent.logger.info(f'Starting the testing for run {run_number+1}')
                print(f'Starting the testing for run {run_number+1}')

                # To keep statistics of each episode
                wins = 0
                detected = 0
                returns = []
                num_steps = []
                num_win_steps = []
                num_detected_steps = []

                # Log the experiment name as a tag
                mlflow.set_tag("experiment_name", experiment_name)
                # Log notes or additional information
                mlflow.set_tag("notes", "This is a test run")
                # Log run number
                mlflow.set_tag("run_number", run_number)

                for episode in range(1, args.episodes + 1):
                    # Reset
                    observation = agent.request_game_reset()
                    # Select action
                    action = agent.select_action(observation)
                    # Act/Move
                    observation = agent.make_step(action)

                    state = observation.state
                    reward = observation.reward
                    done = observation.done
                    info = observation.info
                    steps = observation.steps

                    if observation.info['end_reason'] == 'detected':
                        detection = True
                    elif observation.info['end_reason'] == 'goal_reach':
                        win = True

                    if win:
                        wins += 1
                        num_win_steps += [steps]
                    if detection:
                        detected +=1
                        num_detected_steps += [steps]
                    # Get the return of the whole episdo by sum rewards
                    returns += [reward]
                    num_steps += [steps]

                    test_win_rate = (wins/episode) * 100
                    test_detection_rate = (detected/episode) * 100
                    test_average_returns = np.mean(returns)
                    test_std_returns = np.std(returns)
                    test_average_episode_steps = np.mean(num_steps)
                    test_std_episode_steps = np.std(num_steps)
                    test_average_win_steps = np.mean(num_win_steps)
                    test_std_win_steps = np.std(num_win_steps)
                    test_average_detected_steps = np.mean(num_detected_steps)
                    test_std_detected_steps = np.std(num_detected_steps)

                    # Log and report every X episodes
                    if episode % args.test_each == 0 and episode != 0:
                        text = f'''Tested after {episode} episodes.
                            Wins={wins},
                            Detections={detected},
                            winrate={test_win_rate:.3f}%,
                            detection_rate={test_detection_rate:.3f}%,
                            average_returns={test_average_returns:.3f} +- {test_std_returns:.3f},
                            average_episode_steps={test_average_episode_steps:.3f} +- {test_std_episode_steps:.3f},
                            average_win_steps={test_average_win_steps:.3f} +- {test_std_win_steps:.3f},
                            average_detected_steps={test_average_detected_steps:.3f} +- {test_std_detected_steps:.3f}
                            '''
                        agent.logger.info(text)
                        print(text)
                        # Store in tensorboard
                        #writer.add_scalar("charts/test_avg_win_rate", test_win_rate, episode)
                        #writer.add_scalar("charts/test_avg_detection_rate", test_detection_rate, episode)
                        #writer.add_scalar("charts/test_avg_returns", test_average_returns , episode)
                        #writer.add_scalar("charts/test_std_returns", test_std_returns , episode)
                        #writer.add_scalar("charts/test_avg_episode_steps", test_average_episode_steps , episode)
                        #writer.add_scalar("charts/test_std_episode_steps", test_std_episode_steps , episode)
                        #writer.add_scalar("charts/test_avg_win_steps", test_average_win_steps , episode)
                        #writer.add_scalar("charts/test_std_win_steps", test_std_win_steps , episode)
                        #writer.add_scalar("charts/test_avg_detected_steps", test_average_detected_steps , episode)
                        #writer.add_scalar("charts/test_std_detected_steps", test_std_detected_steps , episode)
                        # Store in mlflow
                        mlflow.log_metric("test_avg_win_rate", test_win_rate, step=episode)

                        # Log other artifacts or parameters
                        #mlflow.log_param("learning_rate", learning_rate)
                        #mlflow.log_param("batch_size", batch_size)
                
                # Outside episode's for
                # Log the last final episode when it ends
                text = f'''Trial {run_number+1} Final test after {episode} episodes, for {args.episodes} steps.
                    Wins={wins},
                    Detections={detected},
                    winrate={test_win_rate:.3f}%,
                    detection_rate={test_detection_rate:.3f}%,
                    average_returns={test_average_returns:.3f} +- {test_std_returns:.3f},
                    average_episode_steps={test_average_episode_steps:.3f} +- {test_std_episode_steps:.3f},
                    average_win_steps={test_average_win_steps:.3f} +- {test_std_win_steps:.3f},
                    average_detected_steps={test_average_detected_steps:.3f} +- {test_std_detected_steps:.3f}
                    '''

                # Update Statistics for all runs
                trial_win_rate += [test_win_rate]
                trial_detection_rate += [test_detection_rate]
                trial_average_returns += [test_average_returns]
                trial_std_returns += [test_std_returns] 
                trial_average_episode_steps += [test_average_episode_steps] 
                trial_std_episode_steps += [test_std_episode_steps]
                trial_average_win_steps += [test_average_win_steps] 
                trial_std_win_steps += [test_std_win_steps] 
                trial_average_detected_steps += [test_average_detected_steps]
                trial_std_detected_steps += [test_std_detected_steps]
                
            text = f'''Final results after {run_number+1} trials, for {args.episodes} steps.
                    winrate={np.mean(trial_win_rate):.3f}% +- {np.std(trial_win_rate):.3f},
                    detection_rate={np.mean(trial_detection_rate):.3f}% +- {np.std(trial_detection_rate):.3f},
                    average_returns={np.mean(trial_average_returns):.3f} +- {np.std(trial_average_returns):.3f},
                    average_episode_steps={np.mean(trial_average_episode_steps):.3f} +- {np.std(trial_average_episode_steps):.3f},
                    average_win_steps={np.mean(trial_average_win_steps):.3f} +- {np.std(trial_average_win_steps):.3f},
                    average_detected_steps={np.mean(trial_average_detected_steps):.3f} +- {np.std(trial_average_detected_steps):.3f}
                    '''
            agent.logger.info(text)
            print(text)


