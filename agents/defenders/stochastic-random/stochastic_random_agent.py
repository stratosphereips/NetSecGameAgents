# Author: Sebastian Garcia. sebastian.garcia@agents.fel.cvut.cz
# This agents implements a simple probabilistic defender that blocks ip based on probability distribution of the logs

import sys
import logging
import os
import argparse
import numpy as np
import time
from random import choice
from os import path, makedirs
from AIDojoCoordinator.game_components import Action, Observation, ActionType, IP, Network, Service, Data

# This is used so the agent can see the environment and game components
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__) ))))

# With the path fixed, we can import now
from base_agent import BaseAgent
from agent_utils import generate_valid_actions


class StochasticRandomDefenderAgent(BaseAgent):
    def __init__(self, host:str, port:int, role:str, allowed_actions:list, apm_limit:int=None) -> None:
        super().__init__(host, port, role)
        self._allowed_actions = allowed_actions
        self._apm_limit = apm_limit
        if self._apm_limit:
            self.inter_action_interval = 60/apm_limit
        else:
            self.inter_action_interval = 0

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
                # First do the step
                observation = self.make_step(action)
                # Then get the reward
                episodic_returns.append(observation.reward)

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

            self._logger.debug(f'Observation received:{observation}')
            # Get the returns of all the episodes played together
            returns.append(np.sum(episodic_returns))
            self._logger.info(f"Episode {episode} ended with return {np.sum(episodic_returns)}. Mean returns={np.mean(returns)}±{np.std(returns)}")
            # Reset the episode
            observation = self.request_game_reset()

        self._logger.info(f"Final results for {self.__class__.__name__} after {num_episodes} episodes: {np.mean(returns)}±{np.std(returns)}")
        self._logger.info("Terminating interaction")
        self.terminate_connection()
    
    def select_action(self, observation:Observation)->Action:
        """ Select an action based on the allowed actions and the current state """
        valid_actions = generate_valid_actions(observation.state)

        # Filter actions based on the allowed action types
        allowed_actions = filter(lambda action: action.type in self._allowed_actions, valid_actions)
        allowed_actions = [a for a  in allowed_actions] + [Action(ActionType.ResetGame, parameters={})]
        # Choose a random action from the allowed actions
        action = choice([a for a  in allowed_actions])
        return action

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    parser.add_argument("--episodes", help="Sets number of testing episodes", default=1, type=int)
    parser.add_argument("--logdir", help="Folder to store logs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    parser.add_argument("--apm", help="Actions per minute", default=10000, type=int, required=False)
    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logging.basicConfig(filename=os.path.join(args.logdir, "stochastic_random_agent.log"), filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',level=logging.DEBUG)

    # Create agent
    agent = StochasticRandomDefenderAgent(args.host, args.port, "Defender", allowed_actions=[ActionType.FindData, ActionType.BlockIP], apm_limit=args.apm)
    agent.play_game(args.episodes)