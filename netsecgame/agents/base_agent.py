from __future__ import annotations

from abc import ABC
import json
import logging
import socket

from ..game_components import (
    Action,
    ActionType,
    AgentInfo,
    AgentRole,
    GameState,
    GameStatus,
    Observation,
    ProtocolConfig,
)


class BaseAgent(ABC):
    def __init__(self, host: str, port: int, role: AgentRole | str) -> None:
        if isinstance(role, str):
            role = AgentRole.from_string(role)

        self._connection_details = (host, port)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._role = role
        self._socket: socket.socket | None = None

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((host, port))
        except socket.error as error:
            self._logger.error("Socket error: %s", error)
            self._socket = None

        self._logger.info("Agent created")

    def __del__(self):
        if self._socket:
            try:
                self._socket.close()
                self._logger.info("Socket closed")
            except socket.error:
                pass

    @property
    def socket(self) -> socket.socket | None:
        return self._socket

    @property
    def role(self) -> AgentRole:
        return self._role

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def terminate_connection(self) -> None:
        if self._socket:
            try:
                self._socket.close()
                self._socket = None
                self._logger.info("Socket closed")
            except socket.error as error:
                self._logger.error("Error closing socket: %s", error)

    def _send_data(self, sock: socket.socket, message: str) -> None:
        self._logger.debug("Sending: %s", message)
        sock.sendall(message.encode())

    def _receive_data(self, sock: socket.socket) -> tuple[GameStatus, dict, str | None]:
        data = b""
        while True:
            chunk = sock.recv(ProtocolConfig.BUFFER_SIZE)
            if not chunk:
                break
            data += chunk
            if ProtocolConfig.END_OF_MESSAGE in data:
                break

        if ProtocolConfig.END_OF_MESSAGE not in data:
            raise ConnectionError("Unfinished connection.")

        payload = data.replace(ProtocolConfig.END_OF_MESSAGE, b"").decode()
        self._logger.debug("Data received from env: %s", payload)

        data_dict = json.loads(payload)
        status = data_dict.get("status", "")
        observation = data_dict.get("observation", {})
        message = data_dict.get("message")
        return GameStatus.from_string(str(status)), observation, message

    def communicate(self, data: Action) -> tuple[GameStatus, dict, str | None]:
        if not isinstance(data, Action):
            raise ValueError("Incorrect data type! Data should be ONLY of type Action")
        if self._socket is None:
            raise ConnectionError("Agent socket is not connected.")

        self._send_data(self._socket, data.to_json())
        return self._receive_data(self._socket)

    def make_step(self, action: Action) -> Observation | None:
        _, observation_dict, _ = self.communicate(action)
        if observation_dict:
            return Observation(
                GameState.from_dict(observation_dict["state"]),
                observation_dict["reward"],
                observation_dict["end"],
                observation_dict["info"],
            )
        return None

    def register(self) -> Observation | None:
        self._logger.info("Registering agent as %s", self.role)
        status, observation_dict, message = self.communicate(
            Action(
                ActionType.JoinGame,
                parameters={"agent_info": AgentInfo(self.__class__.__name__, self.role.value)},
            )
        )
        if status is GameStatus.CREATED:
            self._logger.info("Registration successful! %s", message)
            return Observation(
                GameState.from_dict(observation_dict["state"]),
                observation_dict["reward"],
                observation_dict["end"],
                message,
            )

        self._logger.error("Registration failed! (status: %s, msg: %s)", status, message)
        return None

    def request_game_reset(
        self,
        request_trajectory: bool = False,
        randomize_topology: bool = True,
        randomize_topology_seed: int | None = None,
    ) -> Observation | None:
        parameters = {
            "request_trajectory": request_trajectory,
            "randomize_topology": randomize_topology,
        }
        if randomize_topology_seed is not None:
            parameters["randomize_topology_seed"] = randomize_topology_seed

        status, observation_dict, message = self.communicate(
            Action(ActionType.ResetGame, parameters=parameters)
        )
        if status:
            return Observation(
                GameState.from_dict(observation_dict["state"]),
                observation_dict["reward"],
                observation_dict["end"],
                message,
            )

        self._logger.error("Reset failed! (status: %s, msg: %s)", status, message)
        return None
