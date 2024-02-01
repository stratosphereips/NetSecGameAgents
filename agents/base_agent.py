# Author: Ondrej Lukas, ondrej.lukas@aic.cvut.cz
# Basic agent class that is to be extended in each agent classes
import sys
import argparse
import logging

from os import path
import os
import socket
import json

# This is used so the agent can see the environment and game components
sys.path.append(path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ) ))
from env.game_components import Action, GameState, Observation


class BaseAgent:
    """
    Author: Ondrej Lukas, ondrej.lukas@aic.cvut.cz
    Basic agent for the network based NetSecGame environment. Implemenets communication with the game server.
    """

    def __init__(self, host, port, role:str)->None:
        self._connection_details = (host, port)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._role = role
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((host, port))
        except socket.error as e:
            self._logger.error("Socket error: {e}")
            self.sock = None
        self._logger.info("Agent created")
    
    def __del__(self):
        "In case the extending class did not close the connection, terminate the socket when the object is deleted."
        if self._socket:
            try:
                self._socket.close()
                self._logger.info("Socket closed")
            except socket.error as e:
                print(f"Error closing socket: {e}")
    
    def terminate_connection(self):
        "Method for graceful termination of connection. Should be used by any class extending the BaseAgent."
        if self._socket:
            try:
                self._socket.close()
                self._socket = None
                self._logger.info("Socket closed")
            except socket.error as e:
                print(f"Error closing socket: {e}")
    @property
    def socket(self)->socket.socket:
        return self._socket
    
    @property
    def role(self)->str:
        return self._role
    
    @property
    def logger(self)->logging.Logger:
        return self._logger
    
    def make_step(self, action: Action)->Observation:
        """
        Method for sendind agent's action to the server and receiving and parsing response into new observation.
        """
        _, observation_dict, _ = self.communicate(action)
        if observation_dict:
            return Observation(GameState.from_json(observation_dict["state"]), observation_dict["reward"], observation_dict["end"],{})
        else:
            return None
    
    def communicate(self, data:object)-> tuple:
        """Method for a data exchange with the server. Expect Action, dict or string as input.
        Outputs tuple with server's response in format (status_dict, response_body, message)"""
        def _send_data(socket, data:str)->None:
            try:
                self._logger.debug(f'Sending: {data}')
                self.socket.sendall(data.encode())
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
        
        _send_data(self._socket, data)
        return _receive_data(self._socket)

    def register(self)->Observation:
        """
        Method for registering agent to the game server.
        Classname is used as agent name and the role is based on the 'role' argument.
        TO BE MODIFIED IN FUTURE WHEN THE GAME SUPPORTS 1 MESSAGE REGISTRATION
        """
        try:
            self._logger.info(f'Registering agent as {self.role}')
            status, observation_dict, message = self.communicate("")
            if 'Insert your nick' in message:
                status, observation_dict, message  = self.communicate({'PutNick': self.__class__.__name__} )
                if 'Which side are' in message:
                    status, observation_dict, message  = self.communicate({'ChooseSide': self.role})
                    if status:
                        self._logger.info('\tRegistration successful')
                        return Observation(GameState.from_json(observation_dict["state"]), observation_dict["reward"], observation_dict["end"],{})
                    else:
                        self._logger.error(f'\tRegistration failed! (status: {status}, msg:{message}')
                        return None
        except Exception as e:
            self._logger.error(f'Exception in register(): {e}')

    def request_game_reset(self)->Observation:
        """
        Method for requesting restart of the game.
        """
        self._logger.debug("Requesting game reset")
        status, observation_dict, message = self.communicate({"Reset":True})
        if status:
            self._logger.debug('\tReset successful')
            return Observation(GameState.from_json(observation_dict["state"]), observation_dict["reward"], observation_dict["end"],{})
        else:
            self._logger.error(f'\rReset failed! (status: {status}, msg:{message}')
            return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    
    args = parser.parse_args()
    log_filename = os.path.dirname(os.path.abspath(__file__)) + '/base_agent.log'
    logging.basicConfig(filename=log_filename, filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s',  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    agent = BaseAgent(args.host, args.port, role="Attacker")
    print(agent.register())
