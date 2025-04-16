# Author: Ondrej Lukas, ondrej.lukas@aic.cvut.cz
# Basic agent class that is to be extended in each agent classes
import sys
import logging

from os import path
import socket
import json
from abc import ABC 

from AIDojoCoordinator.game_components import Action, GameState, Observation, ActionType, GameStatus, AgentInfo, ProtocolConfig

class BaseAgent(ABC):
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
            self._logger.error(f"Socket error: {e}")
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
            return Observation(GameState.from_dict(observation_dict["state"]), observation_dict["reward"], observation_dict["end"], observation_dict["info"])
        else:
            return None
    
    def communicate(self, data:Action)-> tuple:
        """Method for a data exchange with the server. Expect Action, dict or string as input.
        Outputs tuple with server's response in format (status_dict, response_body, message)"""
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
            data = b""  # Initialize an empty byte string

            while True:
                chunk = socket.recv(ProtocolConfig.BUFFER_SIZE)  # Receive a chunk
                if not chunk:  # If no more data, break (connection closed)
                    break
                data += chunk
                if ProtocolConfig.END_OF_MESSAGE in data:  # Check if EOF marker is present
                    break
            if ProtocolConfig.END_OF_MESSAGE not in data:
                raise ConnectionError("Unfinished connection.")
            data = data.replace(ProtocolConfig.END_OF_MESSAGE, b"")  # Remove EOF marker
            data = data.decode() 
            self._logger.debug(f"Data received from env: {data}")
            # extract data from string representation
            data_dict = json.loads(data)
            # Add default values if dict keys are missing
            status = data_dict["status"] if "status" in data_dict else {}
            observation = data_dict["observation"] if "observation" in data_dict else {}
            message = data_dict["message"] if "message" in data_dict else None

            return GameStatus.from_string(status), observation, message
        
        if isinstance(data, Action):
            data = data.to_json()
        else:
            raise ValueError("Incorrect data type! Data should be ONLY of type Action")
        
        _send_data(self._socket, data)
        return _receive_data(self._socket)
    
    def register(self)->Observation:
        """
        Method for registering agent to the game server.
        Classname is used as agent name and the role is based on the 'role' argument.
        """
        try:
            self._logger.info(f'Registering agent as {self.role}')
            status, observation_dict, message = self.communicate(Action(ActionType.JoinGame,
                                                                         parameters={"agent_info":AgentInfo(self.__class__.__name__,self.role)}))
            self._logger.info(f'\tRegistering agent as {status, observation_dict, message}')
            if status is GameStatus.CREATED:
                self._logger.info(f"\tRegistration successful! {message}")
                return Observation(GameState.from_dict(observation_dict["state"]), observation_dict["reward"], observation_dict["end"], message)
            else:
                self._logger.error(f'\tRegistration failed! (status: {status}, msg:{message}')
                return None
        except Exception as e:
            self._logger.error(f'Exception in register(): {e}')
    
    def request_game_reset(self, request_trajectory=False)->Observation:
        """
        Method for requesting restart of the game.
        """
        self._logger.debug("Requesting game reset")
        status, observation_dict, message = self.communicate(Action(ActionType.ResetGame, parameters={"request_trajectory":request_trajectory}))
        if status:
            self._logger.debug('\tReset successful')
            return Observation(GameState.from_dict(observation_dict["state"]), observation_dict["reward"], observation_dict["end"], message)
        else:
            self._logger.error(f'\rReset failed! (status: {status}, msg:{message}')
            return None
