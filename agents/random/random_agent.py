#Author: Ondrej Lukas, ondrej.lukas@aic.cvut.cz
# This agents just randomnly picks actions. No learning
import sys
import logging
import os
from random import choice
import argparse
import numpy as np
#from torch.utils.tensorboard import SummaryWriter

# This is used so the agent can see the environment and game components
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ) ) )))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) )))
# with the path fixed, we can import now
from env.game_components import Action, Observation, GameState
from base_agent import BaseAgent
from agent_utils import generate_valid_actions


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
                # select the action randomly
                action = self.select_action(observation)
                episodic_returns.append(observation.reward)
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
    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logging.basicConfig(filename=os.path.join(args.logdir, "random_agent.log"), filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',level=logging.INFO)

    # Setup tensorboard
    #run_name = f"netsecgame__llm__{env.seed}__{int(time.time())}"

    # Create agent
    agent = RandomAgent(args.host, args.port,"Attacker", seed=42)
    agent.play_game(args.episodes)

    # trial_win_rate = []
    # trial_detection_rate = []
    # trial_average_returns = []
    # trial_std_returns = []
    # trial_average_episode_steps = []
    # trial_std_episode_steps = []
    # trial_average_win_steps = []
    # trial_std_win_steps = []
    # trial_average_detected_steps = []
    # trial_std_detected_steps = []

    # for j in range(args.num_trials):
    #     writer = SummaryWriter(f"agents/random/logs/{run_name}_j")
    #     # Testing
    #     wins = 0
    #     detected = 0
    #     returns = []
    #     num_steps = []
    #     num_win_steps = []
    #     num_detected_steps = []
    #     logger.info(f'Starting the testing for run {j+1}')
    #     print(f'Starting the testing for run {j+1}')
    #     for episode in range(1, args.episodes + 1):
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

    #         test_win_rate = (wins/episode) * 100
    #         test_detection_rate = (detected/episode) * 100
    #         test_average_returns = np.mean(returns)
    #         test_std_returns = np.std(returns)
    #         test_average_episode_steps = np.mean(num_steps)
    #         test_std_episode_steps = np.std(num_steps)
    #         test_average_win_steps = np.mean(num_win_steps)
    #         test_std_win_steps = np.std(num_win_steps)
    #         test_average_detected_steps = np.mean(num_detected_steps)
    #         test_std_detected_steps = np.std(num_detected_steps)


    #         if episode % args.test_each == 0 and episode != 0:
    #             print(f'Episode {episode}')
    #             logger.info(f'Episode {episode}')
    #             text = f'''Tested after {episode} episodes.
    #                 Wins={wins},
    #                 Detections={detected},
    #                 winrate={test_win_rate:.3f}%,
    #                 detection_rate={test_detection_rate:.3f}%,
    #                 average_returns={test_average_returns:.3f} +- {test_std_returns:.3f},
    #                 average_episode_steps={test_average_episode_steps:.3f} +- {test_std_episode_steps:.3f},
    #                 average_win_steps={test_average_win_steps:.3f} +- {test_std_win_steps:.3f},
    #                 average_detected_steps={test_average_detected_steps:.3f} +- {test_std_detected_steps:.3f}
    #                 '''
    #             logger.info(text)
    #             print(text)
    #             # Store in tensorboard
    #             writer.add_scalar("charts/test_avg_win_rate", test_win_rate, episode)
    #             writer.add_scalar("charts/test_avg_detection_rate", test_detection_rate, episode)
    #             writer.add_scalar("charts/test_avg_returns", test_average_returns , episode)
    #             writer.add_scalar("charts/test_std_returns", test_std_returns , episode)
    #             writer.add_scalar("charts/test_avg_episode_steps", test_average_episode_steps , episode)
    #             writer.add_scalar("charts/test_std_episode_steps", test_std_episode_steps , episode)
    #             writer.add_scalar("charts/test_avg_win_steps", test_average_win_steps , episode)
    #             writer.add_scalar("charts/test_std_win_steps", test_std_win_steps , episode)
    #             writer.add_scalar("charts/test_avg_detected_steps", test_average_detected_steps , episode)
    #             writer.add_scalar("charts/test_std_detected_steps", test_std_detected_steps , episode)


    #     text = f'''Trial {j+1} Final test after {episode} episodes, for {args.episodes} steps.
    #         Wins={wins},
    #         Detections={detected},
    #         winrate={test_win_rate:.3f}%,
    #         detection_rate={test_detection_rate:.3f}%,
    #         average_returns={test_average_returns:.3f} +- {test_std_returns:.3f},
    #         average_episode_steps={test_average_episode_steps:.3f} +- {test_std_episode_steps:.3f},
    #         average_win_steps={test_average_win_steps:.3f} +- {test_std_win_steps:.3f},
    #         average_detected_steps={test_average_detected_steps:.3f} +- {test_std_detected_steps:.3f}
    #         '''
        
    #     trial_win_rate += [test_win_rate]
    #     trial_detection_rate += [test_detection_rate]
    #     trial_average_returns += [test_average_returns]
    #     # trial_std_returns += [test_std_returns] 
    #     trial_average_episode_steps += [test_average_episode_steps] 
    #     # trial_std_episode_steps += [test_std_episode_steps]
    #     trial_average_win_steps += [test_average_win_steps] 
    #     # trial_std_win_steps += [test_std_win_steps] 
    #     trial_average_detected_steps += [test_average_detected_steps]
    #     # trial_std_detected_steps += [test_std_detected_steps]
        
    # text = f'''Final results after {j+1} trials, for {args.episodes} steps.
    #         winrate={np.mean(trial_win_rate):.3f}% +- {np.std(trial_win_rate):.3f},
    #         detection_rate={np.mean(trial_detection_rate):.3f}% +- {np.std(trial_detection_rate):.3f},
    #         average_returns={np.mean(trial_average_returns):.3f} +- {np.std(trial_average_returns):.3f},
    #         average_episode_steps={np.mean(trial_average_episode_steps):.3f} +- {np.std(trial_average_episode_steps):.3f},
    #         average_win_steps={np.mean(trial_average_win_steps):.3f} +- {np.std(trial_average_win_steps):.3f},
    #         average_detected_steps={np.mean(trial_average_detected_steps):.3f} +- {np.std(trial_average_detected_steps):.3f}
    #         '''
    # logger.info(text)
    # print(text)
