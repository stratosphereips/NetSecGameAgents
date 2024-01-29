# Author: Ondrej Lukas, ondrej.lukas@aic.cvut.cz
# Basic agent class that is to be extended in each agent classes
import sys
import argparse
import logging

from os import path
import os
import socket
import json
import numpy as np
# This is used so the agent can see the environment and game components
sys.path.append(path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ) ))
from env.game_components import Action, GameState, Observation


class BaseAgent:
    """
    Author: Ondrej Lukas, ondrej.lukas@aic.cvut.cz
    Basic agent for the network based NetSecGame environment. Implemenets communication with the game server.
    """

    def __init__(self, host, port)->None:
        self._connection_details = (host, port)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info("Agent created")
    
    def step(self, observation: Observation)->Action:
       """
       Method for generating action based on the given observation of the environment
       """
       raise NotImplementedError
    
    def communicate(self, socket:socket.socket, data:object)-> dict:
        """Method for a data exchange with the server. Expect Action, dict or string as input.
        Outputs dictionary with server's response"""
        def _send_data(socket, data:str)->None:
            try:
                self._logger.debug(f'Sending: {data}')
                socket.sendall(data.encode())
            except Exception as e:
                self._logger.error(f'Exception in _send_data(): {e}')
                raise e
            
        def _receive_data(socket)->tuple:
            """
            Receive data from server
            """
            # Receive data from the server
            data = socket.recv(8192).decode()
            self._logger.debug(f"Data received from env: {data}")
            # extract data from string representation
            data_dict = json.loads(data)
            # Add default values if dict keys are missing
            status = data_dict["status"] if "status" in data_dict else {}
            observation = json.loads(data_dict["observation"]) if "observation" in data_dict else {}
            message = data_dict["message"] if "message" in data_dict else None

            return status, observation, message
        
        if isinstance(data, Action):
            data = data.as_json()
        elif isinstance(data, dict):
            data = json.dumps(data)
        elif type(data) is not str:
            raise ValueError("Incorrect data type! Supported types are 'Action', dict, and str.")
        
        _send_data(socket, data)
        return _receive_data(socket)

    def register(self,socket, role="Attacker"):
        """
        Method for registering agent to the game server.
        Classname is used as agent name and the role is based on the 'role' argument.
        TO BE REMOVED IN FUTURE WHEN THE GAME SUPPORTS 1 MESSAGE REGISTRATION
        """
        try:
            self._logger.info(f'Registering agent as {role}')
            status, observation, message = self.communicate(socket,"")
            if 'Insert your nick' in message:
                status, observation, message  = self.communicate(socket, {'PutNick': self.__class__.__name__} )
                if 'Which side are' in message:
                    status, observation, message  = self.communicate(socket, {'ChooseSide': role})
                    if status:
                        self._logger.info('\tRegistration successful')
                        return status, observation, message
                    else:
                        self._logger.error('\tRegistration failed!')
                        return False
        except Exception as e:
            self._logger.error(f'Exception in register(): {e}')

    def play_game(self, num_episodes=1):
        """
        The main function for the gameplay. Handles socket opening and closing, agent registration and the main interaction loop.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as game_socket:
            game_socket.connect(self._connection_details)
            # Register
            status, observation_dict, message = self.register(game_socket, role="Attacker")
            returns = []
            for episode in range(num_episodes):
                while observation_dict and not observation_dict["end"]:
                    self._logger.info(f'Observation received:{observation_dict}')
                    # Convert the state in observation from json string to dict
                    action = self.step(Observation(GameState.from_json(observation_dict["state"]), observation_dict["reward"], observation_dict["end"],{}))
                    status, observation_dict, message = self.communicate(game_socket, action)
                self._logger.info(f'Observation received:{observation_dict}')
                returns.append(observation_dict["reward"])
                self._logger.info(f"Episode {episode} ended. Mean returns={np.mean(returns)}±{np.std(returns)}")

        self._logger.info(f"Final results for {self.__class__.__name__} after {num_episodes} episodes: {np.mean(returns)}±{np.std(returns)}")
        self._logger.info("Terminating interaction")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    
    args = parser.parse_args()
    log_filename = os.path.dirname(os.path.abspath(__file__)) + '/base_agent.log'
    logging.basicConfig(filename=log_filename, filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s',  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    agent = BaseAgent(args.host, args.port)
    agent.play_game()