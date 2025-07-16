# Author: Ondrej Lukas, ondrej.lukas@aic.cvut.cz
# Basic agent class that is to be extended in each agent classes
import logging
import socket
import json
from abc import ABC 

from AIDojoCoordinator.game_components import Action, GameState, Observation, ActionType, GameStatus, AgentInfo, ProtocolConfig

class BaseAgent(ABC):
    """
    Base agent class for the NetSecGame environment.
    This class implements the basic functionality for communication with the game server.
    It provides methods for sending actions, receiving observations, and managing the connection.
    This class should be extended by specific agent implementations.
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
    
    def __del__(self)->None:
        """
        Destructor to ensure the socket connection is closed when the object is deleted.

        If the extending class does not explicitly close the socket connection,
        this method attempts to close it and logs the action. Any socket errors
        encountered during closure are logged.

        Returns:
            None
        """
        "In case the extending class did not close the connection, terminate the socket when the object is deleted."
        if self._socket:
            try:
                self._socket.close()
                self._logger.info("Socket closed")
            except socket.error as e:
                self._logger.error(f"Error closing socket: {e}")
    
    def terminate_connection(self):
        """
        Gracefully terminates the network connection by closing the socket.

        This method should be used by any class extending the BaseAgent to ensure
        proper cleanup of the socket resource. If the socket is open, it attempts
        to close it and logs the action. If an error occurs during closing, it
        logs the error message.

        Raises:
            socket.error: If an error occurs while closing the socket.
        """
        "Method for graceful termination of connection. Should be used by any class extending the BaseAgent."
        if self._socket:
            try:
                self._socket.close()
                self._socket = None
                self._logger.info("Socket closed")
            except socket.error as e:
                self._logger.error(f"Error closing socket: {e}")
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
        Sends the agent's action to the server and receives the response, parsing it into a new observation.

        Args:
            action (Action): The action to send to the server.

        Returns:
            Observation: The observation received from the server after performing the action.
        """
        _, observation_dict, _ = self.communicate(action)
        if observation_dict:
            return Observation(GameState.from_dict(observation_dict["state"]), observation_dict["reward"], observation_dict["end"], observation_dict["info"])
        else:
            return None
    
    def communicate(self, data:Action)-> tuple:
        """
        Exchanges data with the server.

        Args:
            data (Action): The action to send to the server. Must be of type Action.

        Returns:
            tuple: A tuple containing:
            - GameStatus: The status of the game returned by the server.
            - dict: The observation dictionary from the server.
            - str or None: An optional message from the server.

        Raises:
            ValueError: If the input data is not of type Action.
            ConnectionError: If the connection is unfinished or data is incomplete.
            Exception: If sending or receiving data fails.
        """
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
        Registers the agent with the game server.

        The agent's class name is used as the agent name, and the role is specified by the 'role' argument.

        Returns:
            Observation: The initial observation received after registration, or None if registration fails.
        """
        try:
            self._logger.info(f'Registering agent as {self.role}')
            status, observation_dict, message = self.communicate(Action(ActionType.JoinGame,
                                                                         parameters={"agent_info":AgentInfo(self.__class__.__name__,self.role)}))
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
        Requests a restart of the game.

        Args:
            request_trajectory (bool): If True, requests the game trajectory to be returned.

        Returns:
            Observation: The initial observation after the game reset, or None if reset fails.
        """
        self._logger.debug("Requesting game reset")
        status, observation_dict, message = self.communicate(Action(ActionType.ResetGame, parameters={"request_trajectory":request_trajectory}))
        if status:
            self._logger.debug('\tReset successful')
            return Observation(GameState.from_dict(observation_dict["state"]), observation_dict["reward"], observation_dict["end"], message)
        else:
            self._logger.error(f'\rReset failed! (status: {status}, msg:{message}')
            return None
