# Author: Ondrej Lukas, ondrej.lukas@aic.cvut.cz
# Basic agent class that is to be extended in each agent classes
import sys
import argparse
import logging

from os import path
import socket
import json
from abc import ABC


# This is used so the agent can see the environment and game components
sys.path.append(path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ) ))
from env.game_components import Action, GameState, Observation, ActionType, GameStatus,AgentInfo, IP, Network, Data
from agent_utils import state_as_ordered_string

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
            data = socket.recv(8192).decode()
            self._logger.debug(f"Data received from env: {data}")
            # extract data from string representation
            data_dict = json.loads(data)
            # Add default values if dict keys are missing
            status = data_dict["status"] if "status" in data_dict else {}
            observation = data_dict["observation"] if "observation" in data_dict else {}
            message = data_dict["message"] if "message" in data_dict else None

            return GameStatus.from_string(status), observation, message
        
        if isinstance(data, Action):
            data = data.as_json()
            print(data)
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
    
    def request_game_reset(self)->Observation:
        """
        Method for requesting restart of the game.
        """
        self._logger.debug("Requesting game reset")
        status, observation_dict, message = self.communicate(Action(ActionType.ResetGame, {}))
        if status:
            self._logger.debug('\tReset successful')
            return Observation(GameState.from_dict(observation_dict["state"]), observation_dict["reward"], observation_dict["end"], message)
        else:
            self._logger.error(f'\rReset failed! (status: {status}, msg:{message}')
            return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    
    args = parser.parse_args()
    log_filename = path.dirname(path.abspath(__file__)) + '/base_agent.log'
    logging.basicConfig(filename=log_filename, filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s',  datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
   
    # # #######################################
    # # ATTACKER X ATTACKER 
    # # #######################################
    
    # #register both agents
    # agent1 = BaseAgent(args.host, args.port, role="Attacker")
    # obs1 = agent1.register()
    # agent2 = BaseAgent(args.host, args.port, role="Attacker")
    # obs2 = agent2.register()

    # # network scans - both agents
    # obs1 = agent1.make_step(Action(
    #     ActionType.ScanNetwork,
    #     params={
    #         "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
    #         "target_network":Network("192.168.1.0", 24)
    #         }
    #     )
    # )
    # obs2 = agent2.make_step(Action(
    #     ActionType.ScanNetwork,
    #     params={
    #         "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs2.state.controlled_hosts))[0],
    #         "target_network":Network("192.168.1.0", 24)
    #         }
    #     )
    # )
    # # service scans - both agents
    # obs1 = agent1.make_step(Action(
    #     ActionType.FindServices,
    #     params={
    #         "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
    #         "target_host":IP("192.168.1.2")
    #         }
    #     )
    # )
    # obs2 = agent2.make_step(Action(
    #     ActionType.FindServices,
    #     params={
    #         "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs2.state.controlled_hosts))[0],
    #         "target_host":IP("192.168.1.2")
    #         }
    #     )
    # )
    # # agent 1 exploit
    # host, services  = [(k,v) for k,v in obs1.state.known_services.items()][0]
    # obs1 = agent1.make_step(Action(
    #     ActionType.ExploitService,
    #     params={
    #         "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
    #         "target_host":host,
    #         "target_service":list(services)[0]
    #         }
    #     )
    # )
    # # agent 1 find data
    # obs1 = agent1.make_step(Action(
    #     ActionType.FindData,
    #     params={
    #         "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
    #         "target_host":host,
    #         }
    #     )
    # )

    # # agent 1 exfiltrate data to C&C
    # obs1 = agent1.make_step(Action(
    #     ActionType.ExfiltrateData,
    #     params={
    #         "source_host": host,
    #         "target_host": IP("213.47.23.195"),
    #         "data": Data("User1","DataFromServer1")
    #         }
    #     )
    # )

    # # agent 2 find data
    # obs2 = agent2.make_step(Action(
    #     ActionType.FindData,
    #     params={
    #         "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs2.state.controlled_hosts))[0],
    #         "target_host":host,
    #         }
    #     )
    # )
    # # agent 2 find data in C&C
    # obs2 = agent2.make_step(Action(
    #     ActionType.FindData,
    #     params={
    #         "source_host": IP("213.47.23.195"),
    #         "target_host": IP("213.47.23.195"),
    #         }
    #     )
    # )
    # print(obs1)
    # print(obs2)


    ########################################
    # ATTACKER X DEFENDER 
    ########################################

    # # register both agents
    # agent1 = BaseAgent(args.host, args.port, role="Attacker")
    # obs1 = agent1.register()
    # agent2 = BaseAgent(args.host, args.port, role="Defender")
    # obs2 = agent2.register()

    # # attacker - scan net
    # obs1 = agent1.make_step(Action(
    #     ActionType.ScanNetwork,
    #     params={
    #         "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
    #         "target_network":Network("192.168.1.0", 24)
    #         }
    #     )
    # )
    
    # # attacker - scan services
    # obs1 = agent1.make_step(Action(
    #     ActionType.FindServices,
    #     params={
    #         "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
    #         "target_host":IP("192.168.1.2")
    #         }
    #     )
    # )
    
    # # agent 1 exploit
    # host, services  = [(k,v) for k,v in obs1.state.known_services.items()][0]
    # obs1 = agent1.make_step(Action(
    #     ActionType.ExploitService,
    #     params={
    #         "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
    #         "target_host":host,
    #         "target_service":list(services)[0]
    #         }
    #     )
    # )
    # # agent 1 find data
    # obs1 = agent1.make_step(Action(
    #     ActionType.FindData,
    #     params={
    #         "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
    #         "target_host":host,
    #         }
    #     )
    # )
    # # agent 2 block connection to C&C
    # obs2 = agent2.make_step(Action(
    #     ActionType.BlockIP,
    #     params={
    #         "source_host": host,
    #         "target_host": host,
    #         "blocked_host": IP("213.47.23.195")
    #         }
    #     )
    # )
    # # agent 1 exfiltrate data to C&C
    # obs1 = agent1.make_step(Action(
    #     ActionType.ExfiltrateData,
    #     params={
    #         "source_host": host,
    #         "target_host": IP("213.47.23.195"),
    #         "data": Data("User1","DataFromServer1")
    #         }
    #     )
    # )

    # print(obs1)
    # print(obs2)

    # #######################################
    # 1 ATTACKER 
    # #######################################
    
    #register both agents
    agent1 = BaseAgent(args.host, args.port, role="Attacker")
    obs1 = agent1.register()
    print(obs1)
    print("----------------------------")
    import time
    time.sleep(1)
    # network scans - both agents
    obs1 = agent1.make_step(Action(
        ActionType.ScanNetwork,
        parameters={
            "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
            "target_network":Network("192.168.0.1", 24) 
            }
        )
    )
    print(obs1)
    print("----------------------------")
    time.sleep(1)
    obs1 = agent1.make_step(Action(
        ActionType.FindServices,
        parameters={
            "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
            "target_host": IP("192.168.0.1")
            }
        )
    )
    print(obs1)
    print("----------------------------")
    obs2 = agent1.request_game_reset()
    print(obs2)
    print("----------------------------")

    # # network scans - both agents
    # obs1 = agent1.make_step(Action(
    #     ActionType.ScanNetwork,
    #     params={
    #         "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
    #         "target_network":Network("192.168.0.1", 24) 
    #         }
    #     )
    # )
    # print(obs1)
    # print("----------------------------")
    # # service scans - both agents
    # obs1 = agent1.make_step(Action(
    #     ActionType.FindServices,
    #     params={
    #         "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
    #         "target_host":IP("192.168.1.2")
    #         }
    #     )
    # )
    # print("----------------------------")
    # # agent 1 exploit
    # host, services  = [(k,v) for k,v in obs1.state.known_services.items()][0]
    # obs1 = agent1.make_step(Action(
    #     ActionType.ExploitService,
    #     params={
    #         "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
    #         "target_host":host,
    #         "target_service":list(services)[0]
    #         }
    #     )
    # )
    # print("----------------------------")
    # # agent 1 find data
    # obs1 = agent1.make_step(Action(
    #     ActionType.FindData,
    #     params={
    #         "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
    #         "target_host":host,
    #         }
    #     )
    # )
    # print("----------------------------")
    # # agent 1 exfiltrate data to C&C
    # obs1 = agent1.make_step(Action(
    #     ActionType.ExfiltrateData,
    #     params={
    #         "source_host": host,
    #         "target_host": IP("213.47.23.195"),
    #         "data": Data("User1","DataFromServer1")
    #         }
    #     )
    # )
    # print(obs1)

    
    # host = list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0]
    # obs1 = agent1.make_step(Action(
    #     ActionType.BlockIP,
    #     params={
    #         "source_host": host,
    #         "target_host": host,
    #         "blocked_host": IP("1.1.1.1")
    #     })
    # )
    # print("----------------------------")
    # obs1 = agent1.make_step(Action(
    #     ActionType.BlockIP,
    #     params={
    #         "source_host": host,
    #         "target_host": host,
    #         "blocked_host": IP("1.1.1.2")
    #     })
    # )
    
    
    
    
    # # # agent 1 exfiltrate data to C&C
    # # obs1 = agent1.make_step(Action(
    # #     ActionType.ExfiltrateData,
    # #     params={
    # #         "source_host": host,
    # #         "target_host": IP("213.47.23.195"),
    # #         "data": Data("User1","DataFromServer1")
    # #         }
    # #     )
    # # )

    # print(obs1)
    # print("##########")
    # print(state_as_ordered_string(obs1.state))





    # obs1 = agent1.request_game_reset()
    # #obs2 = agent2.request_game_reset()
    # print("------------------")
    # print(obs1)
    # #print(obs2)
    # # #obs = agent.request_game_reset()
    # # for ip in obs.state.controlled_hosts:
    # #     if ip != IP("213.47.23.195"):
    # #         src = ip
    # #         break
    # # agent.make_step(Action(ActionType.ScanNetwork, params={"source_host":src, "target_network":Network("192.168.1.0", 24)}))
    # # obs = agent.make_step(Action(ActionType.FindServices, params={"source_host":src, "target_host":IP("192.168.1.2")}))
    # # host, services  = [(k,v) for k,v in obs.state.known_services.items()][0]
    # # service = list(services)[0]
    # # agent.make_step(Action(ActionType.ExploitService, params={"source_host":src, "target_host":host, "target_service":service}))
    # # agent.make_step(Action(action_type=ActionType.FindData, params={"source_host":host, "target_host":host}))
    # # obs = agent.make_step(Action(action_type=ActionType.ExfiltrateData, params={
    # #     "source_host":host,
    # #     "target_host":IP("213.47.23.195"),
    # #     "data": Data("User1","DataFromServer1")
    # #     }
    # # ))
    # # print(obs)
    # # # print("----------------------------")
    # # # obs = agent.request_game_reset()
    # # # print('---------------------------')
    # # # obs = agent.make_step(Action(action_type=ActionType.ExfiltrateData, params={
    # # #     "source_host":host,
    # # #     "target_host":IP("213.47.23.195"),
    # # #     "data": Data("User1","DataFromServer1")
    # # #     }
    # # # ))
    # # # print(obs)

