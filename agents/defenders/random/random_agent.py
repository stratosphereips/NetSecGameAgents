# Author: Sebastian Garcia. sebastian.garcia@agents.fel.cvut.cz
# This agents just randomnly blocks IP. 
import logging
import os
import argparse
import numpy as np
import time
import mlflow

from random import uniform, choice
from AIDojoCoordinator.game_components import Action, Observation, ActionType
from NetSecGameAgents.agents.heuristic_exploration_base_agent import HeuristicExplorationBaseAgent
from NetSecGameAgents.agents.agent_utils import generate_valid_actions


class RandomDefenderAgent(HeuristicExplorationBaseAgent):
    def __init__(self, host:str, port:int, role:str, allowed_actions:list, apm_limit:int=None) -> None:
        super().__init__(host, port, role)
        self._allowed_actions = allowed_actions
        self._apm_limit = apm_limit
        if self._apm_limit:
            self.interval = 60/apm_limit
        else:
            self.interval = 0

    def play_game(self, observation, num_episodes=1):
        """
        The main function for the gameplay. Handles agent registration and the main interaction loop.
        """
        
        returns = []
        num_steps = 0
        for episode in range(num_episodes):
            self._logger.info(f"Playing episode {episode}")
            episodic_returns = []
            start_time = time.time()
            while observation and not observation.end:
                num_steps += 1
                self._logger.debug(f'Observation received:{observation}')
                # Store return in episode
                episodic_returns.append(observation.reward)
                # select the action randomly
                action = self.select_action(observation)
                observation = self.make_step(action)
                # To return
                last_observation = observation
                if self._apm_limit:
                    elapsed_time = time.time() - start_time
                    remaining_time = self.interval - elapsed_time
                    if remaining_time > 0:
                        # Add randomness to the interval by multiplying it with a random factor
                        randomized_interval = max(0, remaining_time *uniform(-1, 5))
                        self._logger.debug(f"Waiting for {randomized_interval}s before next action")
                        time.sleep(randomized_interval)
                    start_time = time.time()
            self._logger.debug(f'Observation received:{observation}')
            returns.append(np.sum(episodic_returns))
            self._logger.info(f"Episode {episode} ended with return{np.sum(episodic_returns)}. Mean returns={np.mean(returns)}±{np.std(returns)}")
            # Reset the episode
            observation = self.request_game_reset()
        self._logger.info(f"Final results for {self.__class__.__name__} after {num_episodes} episodes: {np.mean(returns)}±{np.std(returns)}")
        return (last_observation, num_steps)

    def select_action(self, observation:Observation)->Action:
        valid_actions = generate_valid_actions(observation.state)
        # filter actions based on the allowed action types
        allowed_actions = filter(lambda action: action.type in self._allowed_actions, valid_actions)
        allowed_actions = [a for a  in allowed_actions] + [Action(ActionType.ResetGame, parameters={})]
        action = choice([a for a  in allowed_actions])
        return action


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    parser.add_argument("--episodes", help="Sets number of testing episodes", default=1, type=int)
    parser.add_argument("--test_each", help="Evaluate performance during testing every this number of episodes.", default=10, type=int)
    parser.add_argument("--logdir", help="Folder to store logs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    parser.add_argument("--apm", help="Actions per minute", default=1000000, type=int, required=False)
    parser.add_argument("--evaluate", help="Evaluate the agent and report, instead of playing the game only once.", default=True)
    parser.add_argument("--mlflow_url", help="URL for mlflow tracking server. If not provided, mlflow will store locally.", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logging.basicConfig(filename=os.path.join(args.logdir, "defender_random_agent.log"), filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',level=logging.DEBUG)

    # Create agent
    agent = RandomDefenderAgent(args.host, args.port, "Defender", allowed_actions=[ActionType.FindData, ActionType.ExfiltrateData, ActionType.FindServices, ActionType.BlockIP], apm_limit=args.apm)

    print("Agent created. Starting interaction with the game server...")

    if not args.evaluate:
        # Play the normal game
        observation = agent.register()
        print("Agent registered. Starting to play the game...")
        agent.play_game(observation, args.episodes)
        agent._logger.info("Terminating interaction")
        agent.terminate_connection()
    else:
        # Evaluate the agent performance

        # How it works:
        # - Evaluate for several 'episodes' (parameter)
        # - Each episode finishes with: steps played, return, win/lose. Store all
        # - Each episode compute the avg and std of all.
        # - Every X episodes (parameter), report in log and mlflow
        # - At the end, report in log and mlflow and console

        # Mlflow experiment name        
        experiment_name = "Evaluation of Random Defender Agent"
        if args.mlflow_url:
            mlflow.set_tracking_uri(args.mlflow_url)
        mlflow.set_experiment(experiment_name)
        # Register in the game
        observation = agent.register()
        print("Agent registered. Starting evaluation...")
        with mlflow.start_run(run_name=experiment_name) as run:
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

            # Log more things in Mlflow
            mlflow.set_tag("experiment_name", experiment_name)
            # Log notes or additional information
            mlflow.set_tag("notes", "This is an evaluation")
            mlflow.set_tag("episode_number", args.episodes)
            #mlflow.log_param("learning_rate", learning_rate)

            for episode in range(1, args.episodes + 1):
                agent.logger.info(f'Starting the testing for episode {episode}')
                print(f'Starting the testing for episode {episode}')

                # Play the game for one episode
                observation, num_steps = agent.play_game(observation, 1)

                state = observation.state
                reward = observation.reward
                end = observation.end
                info = observation.info

                if observation.info and observation.info['end_reason'] == 'blocked':
                    detected +=1
                    num_detected_steps += [num_steps]
                    num_detected_returns += [reward]
                elif observation.info and observation.info['end_reason'] == 'goal_reached':
                    wins += 1
                    num_win_steps += [num_steps]
                    num_win_returns += [reward]
                elif observation.info and observation.info['end_reason'] == 'max_steps':
                    max_steps += 1
                    num_max_steps_steps += [num_steps]
                    num_max_steps_returns += [reward]

                # Reset the game
                observation = agent.request_game_reset()

                eval_win_rate = (wins/episode) * 100
                eval_detection_rate = (detected/episode) * 100
                eval_average_returns = np.mean(num_detected_returns+num_win_returns+num_max_steps_returns)
                eval_std_returns = np.std(num_detected_returns+num_win_returns+num_max_steps_returns)
                eval_average_episode_steps = np.mean(num_win_steps+num_detected_steps+num_max_steps_steps)
                eval_std_episode_steps = np.std(num_win_steps+num_detected_steps+num_max_steps_steps)
                eval_average_win_steps = np.mean(num_win_steps)
                eval_std_win_steps = np.std(num_win_steps)
                eval_average_detected_steps = np.mean(num_detected_steps)
                eval_std_detected_steps = np.std(num_detected_steps)
                eval_average_max_steps_steps = np.mean(num_max_steps_steps)
                eval_std_max_steps_steps = np.std(num_max_steps_steps)

                # Log and report every X episodes
                if episode % args.test_each == 0 and episode != 0:
                    text = f'''Tested after {episode} episodes.
                        Wins={wins},
                        Detections={detected},
                        winrate={eval_win_rate:.3f}%,
                        detection_rate={eval_detection_rate:.3f}%,
                        average_returns={eval_average_returns:.3f} +- {eval_std_returns:.3f},
                        average_episode_steps={eval_average_episode_steps:.3f} +- {eval_std_episode_steps:.3f},
                        average_win_steps={eval_average_win_steps:.3f} +- {eval_std_win_steps:.3f},
                        average_detected_steps={eval_average_detected_steps:.3f} +- {eval_std_detected_steps:.3f}
                        average_max_steps_steps={eval_std_max_steps_steps:.3f} +- {eval_std_max_steps_steps:.3f},
                        '''
                    agent.logger.info(text)
                    print(text)
                    # Store in mlflow
                    mlflow.log_metric("eval_avg_win_rate", eval_win_rate, step=episode)
                    mlflow.log_metric("eval_avg_detection_rate", eval_detection_rate, step=episode)
                    mlflow.log_metric("eval_avg_returns", eval_average_returns, step=episode)
                    mlflow.log_metric("eval_std_returns", eval_std_returns, step=episode)
                    mlflow.log_metric("eval_avg_episode_steps", eval_average_episode_steps, step=episode)
                    mlflow.log_metric("eval_std_episode_steps", eval_std_episode_steps, step=episode)
                    mlflow.log_metric("eval_avg_win_steps", eval_average_win_steps, step=episode)
                    mlflow.log_metric("eval_std_win_steps", eval_std_win_steps, step=episode)
                    mlflow.log_metric("eval_avg_detected_steps", eval_average_detected_steps, step=episode)
                    mlflow.log_metric("eval_std_detected_steps", eval_std_detected_steps, step=episode)
                    mlflow.log_metric("eval_avg_max_steps_steps", eval_average_max_steps_steps, step=episode)
                    mlflow.log_metric("eval_std_max_steps_steps", eval_std_max_steps_steps, step=episode)

            
            # Log the last final episode when it ends
            text = f'''Episode {episode}. Final eval after {episode} episodes, for {args.episodes} steps.
                Wins={wins},
                Detections={detected},
                winrate={eval_win_rate:.3f}%,
                detection_rate={eval_detection_rate:.3f}%,
                average_returns={eval_average_returns:.3f} +- {eval_std_returns:.3f},
                average_episode_steps={eval_average_episode_steps:.3f} +- {eval_std_episode_steps:.3f},
                average_win_steps={eval_average_win_steps:.3f} +- {eval_std_win_steps:.3f},
                average_detected_steps={eval_average_detected_steps:.3f} +- {eval_std_detected_steps:.3f}
                average_max_steps_steps={eval_std_max_steps_steps:.3f} +- {eval_std_max_steps_steps:.3f},
                '''

            agent.logger.info(text)
            print(text)
            agent._logger.info("Terminating interaction")
            agent.terminate_connection()

            # Print and log the mlflow experiment ID, run ID, and storage location
            experiment_id = run.info.experiment_id
            run_id = run.info.run_id
            storage_location = "locally" if not args.mlflow_url else f"at {args.mlflow_url}"
            print(f"MLflow Experiment ID: {experiment_id}")
            print(f"MLflow Run ID: {run_id}")
            print(f"Experiment saved {storage_location}")
            agent._logger.info(f"MLflow Experiment ID: {experiment_id}")
            agent._logger.info(f"MLflow Run ID: {run_id}")
            agent._logger.info(f"Experiment saved {storage_location}")