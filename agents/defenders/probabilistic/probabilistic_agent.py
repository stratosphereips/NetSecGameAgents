# Author: Sebastian Garcia. sebastian.garcia@agents.fel.cvut.cz
# This agents implements a simple probabilistic defender that blocks ip based on probability distribution of the logs

import sys
import logging
import os
import argparse
import numpy as np
import time
from random import choice
from random import uniform
# This is used so the agent can see the environment and game components
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
)
# with the path fixed, we can import now
from AIDojoCoordinator.game_components import Action, Observation, ActionType
# importing agent utils and base agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ))))
from base_agent import BaseAgent
from agent_utils import generate_valid_actions


class ProbabilisticDefenderAgent(BaseAgent):
    def __init__(self, host:str, port:int, role:str, allowed_actions:list, apm_limit:int=None) -> None:
        super().__init__(host, port, role)
        self._allowed_actions = allowed_actions
        self._apm_limit = apm_limit
        if self._apm_limit:
            self.interval = 60/apm_limit
        else:
            self.interval = 0

    def play_game(self, num_episodes=1):
        """
        The main function for the gameplay. Handles agent registration and the main interaction loop.
        """
        
        observation = self.register()
        returns = []
        for episode in range(num_episodes):
            episodic_returns = []
            start_time = time.time()
            while observation and not observation.end:
                self._logger.debug(f'Observation received:{observation}')
                # select the action randomly
                action = self.select_action(observation)
                episodic_returns.append(observation.reward)
                observation = self.make_step(action)
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
        self._logger.info("Terminating interaction")
        self.terminate_connection()
    
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
    parser.add_argument("--logdir", help="Folder to store logs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    parser.add_argument("--apm", help="Actions per minute", default=10, type=int, required=False)
    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logging.basicConfig(filename=os.path.join(args.logdir, "defender_random_agent.log"), filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',level=logging.DEBUG)

    # Create agent
    agent = ProbabilisticDefenderAgent(args.host, args.port, "Defender", allowed_actions=[ActionType.FindData, ActionType.ExfiltrateData, ActionType.FindServices], apm_limit=args.apm)
    agent.play_game(args.episodes)


"""
From original global defender
class SimplisticDefender:
    def __init__(self, config_file) -> None:
        self.task_config = ConfigParser(config_file)
        self.logger = logging.getLogger('Netsecenv-Defender')
        defender_type = self.task_config.get_defender_type()
        self.logger.info(f"Defender set to be of type '{defender_type}'")
        match defender_type:
            case "NoDefender":
                self._defender_type = None
            case 'StochasticDefender':
                # For now there is only one type of defender
                self._defender_type = "Stochastic"
                self.detection_probability = self._read_detection_probabilities()
            case "StochasticWithThreshold":
                self._defender_type = "StochasticWithThreshold"
                self.detection_probability = self._read_detection_probabilities()
                self._defender_thresholds = self.task_config.get_defender_thresholds()
                self._defender_thresholds["tw_size"] = self.task_config.get_defender_tw_size()
                self._actions_played = []
            case _: # Default option - no defender
                self._defender_type = None
    
    def _read_detection_probabilities(self)->dict:
        # Method to read detection probabilities from the task config task.
        detection_probability = {}
        detection_probability[components.ActionType.ScanNetwork] = self.task_config.read_defender_detection_prob('scan_network')
        detection_probability[components.ActionType.FindServices] = self.task_config.read_defender_detection_prob('find_services')
        detection_probability[components.ActionType.ExploitService] = self.task_config.read_defender_detection_prob('exploit_service')
        detection_probability[components.ActionType.FindData] = self.task_config.read_defender_detection_prob('find_data')
        detection_probability[components.ActionType.ExfiltrateData] = self.task_config.read_defender_detection_prob('exfiltrate_data')
        detection_probability[components.ActionType.BlockIP] = self.task_config.read_defender_detection_prob('exfiltrate_data')
        self.logger.info(f"Detection probabilities:{detection_probability}")
        return detection_probability

    def detect(self, state:components.GameState, action:components.Action, actions_played):
        # Checks if current action was detected based on the defendr type:
        if self._defender_type is not None: # There is a defender present
            match self._defender_type:
                case "Stochastic":
                    detection = self._stochastic_detection(action)
                    self.logger.info(f"\tAction detected?: {detection}")
                    return detection
                case "StochasticWithThreshold":
                    self.logger.info(f"Checking detection based on rules: {action}")
                    detection = self._stochastic_detection_with_thresholds(action, actions_played)
                    self.logger.info(f"\tAction detected?: {detection}")
                    return detection
        else: # No defender in the environment
            logger.info("\tNo defender present")
            return False
    
    def _stochastic_detection_with_thresholds(self, action:components.Action, actions_played)->bool:        
        # Method used for detection with stochastic defender with minimal thresholds
        if len(actions_played) > self._defender_thresholds["tw_size"]: # single action is never detected:
            last_n_actions = actions_played[-self._defender_thresholds["tw_size"]:]
            last_n_action_types = [action.type for action in last_n_actions]
            repeated_action_episode = actions_played.count(action)
            self.logger.info('\tThreshold check')
            # update threh
            match action.type: # thresholds are based on action type
                case components.ActionType.ScanNetwork:
                    tw_ratio = last_n_action_types.count(components.ActionType.ScanNetwork)/self._defender_thresholds["tw_size"]
                    num_consecutive_scans = max(sum(1 for item in grouped if item == components.ActionType.ScanNetwork)
                                                for _,grouped in itertools.groupby(last_n_action_types))
                    if tw_ratio < self._defender_thresholds[components.ActionType.ScanNetwork]["tw_ratio"] and num_consecutive_scans < self._defender_thresholds[components.ActionType.ScanNetwork]["consecutive_actions"]:
                        return False
                    else:
                        self.logger.info(f"\t\t Threshold crossed - TW ratio:{tw_ratio}(T={self._defender_thresholds[components.ActionType.ScanNetwork]['tw_ratio']}), #consecutive actions:{num_consecutive_scans} (T={self._defender_thresholds[components.ActionType.ScanNetwork]['consecutive_actions']})")
                        return self._stochastic_detection(action)
                case components.ActionType.FindServices:
                    tw_ratio = last_n_action_types.count(components.ActionType.FindServices)/self._defender_thresholds["tw_size"]
                    num_consecutive_scans = max(sum(1 for item in grouped if item == components.ActionType.FindServices)
                                                for _,grouped in itertools.groupby(last_n_action_types))
                    if tw_ratio < self._defender_thresholds[components.ActionType.FindServices]["tw_ratio"] and num_consecutive_scans < self._defender_thresholds[components.ActionType.FindServices]["consecutive_actions"]:
                        return False
                    else:
                        self.logger.info(f"\t\t Threshold crossed - TW ratio:{tw_ratio}(T={self._defender_thresholds[components.ActionType.FindServices]['tw_ratio']}), #consecutive actions:{num_consecutive_scans} (T={self._defender_thresholds[components.ActionType.FindServices]['consecutive_actions']})")
                        return self._stochastic_detection(action)
                case components.ActionType.FindData:
                    tw_ratio = last_n_action_types.count(components.ActionType.FindData)/self._defender_thresholds["tw_size"]
                    if tw_ratio < self._defender_thresholds[components.ActionType.FindData]["tw_ratio"] and repeated_action_episode < self._defender_thresholds[components.ActionType.FindData]["repeated_actions_episode"]:
                        return False
                    else:
                        self.logger.info(f"\t\t Threshold crossed - TW ratio:{tw_ratio}(T={self._defender_thresholds[components.ActionType.FindData]['tw_ratio']}), #repeated actions:{repeated_action_episode}")
                        return self._stochastic_detection(action)
                case components.ActionType.ExploitService:
                    tw_ratio = last_n_action_types.count(components.ActionType.ExploitService)/self._defender_thresholds["tw_size"]
                    if tw_ratio < self._defender_thresholds[components.ActionType.ExploitService]["tw_ratio"] and repeated_action_episode < self._defender_thresholds[components.ActionType.ExploitService]["repeated_actions_episode"]:
                        return False
                    else:
                        self.logger.info(f"\t\t Threshold crossed - TW ratio:{tw_ratio}(T={self._defender_thresholds[components.ActionType.ExploitService]['tw_ratio']}), #repeated actions:{repeated_action_episode}")
                        return self._stochastic_detection(action)
                case components.ActionType.ExfiltrateData:
                    tw_ratio = last_n_action_types.count(components.ActionType.ExfiltrateData)/self._defender_thresholds["tw_size"]
                    num_consecutive_scans = max(sum(1 for item in grouped if item == components.ActionType.ExfiltrateData)
                                                for _,grouped in itertools.groupby(last_n_action_types))
                    if tw_ratio < self._defender_thresholds[components.ActionType.ExfiltrateData]["tw_ratio"] and num_consecutive_scans < self._defender_thresholds[components.ActionType.ExfiltrateData]["consecutive_actions"]:
                        return False
                    else:
                        self.logger.info(f"\t\t Threshold crossed - TW ratio:{tw_ratio}(T={self._defender_thresholds[components.ActionType.ExfiltrateData]['tw_ratio']}), #consecutive actions:{num_consecutive_scans} (T={self._defender_thresholds[components.ActionType.ExfiltrateData]['consecutive_actions']})")
                        return self._stochastic_detection(action)
                case _: # default case - No detection
                    return False
        return False
    
    def _stochastic_detection(self, action: components.Action)->bool:
        # Method stochastic detection based on action default probability
        roll = random.random()
        self.logger.info(f"\tRunning stochastic detection. {roll} < {self.detection_probability[action.type]}")
        return roll < self.detection_probability[action.type]
    
    def reset(self)->None:
        self.logger.info("Defender resetted")
"""