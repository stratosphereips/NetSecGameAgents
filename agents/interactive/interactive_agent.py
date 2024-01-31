# Author: Ondrej Lukas, ondrej.lukas@aic.cvut.cz
# This agent allows to manually play the Network Security Game
import sys
import argparse
import logging
import random
from anyio import Path
from termcolor import colored
from os import path
import os
import socket
import json
from enum import Enum
# This is used so the agent can see the environment and game components
sys.path.append(path.dirname(path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ) )))

from env.game_components import Network, IP, Service, Data
from env.game_components import ActionType, Action, GameState, Observation

#log_filename=Path('interactive_agent.log')
log_filename = os.path.dirname(os.path.abspath(__file__)) + '/interactive_agent.log'
logging.basicConfig(filename=log_filename, filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s',  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger('Interactive-agent')
logger.info('Start')

class InputType(Enum):
    """
    What is this class for?
    """
    HOST = 1
    NETWORK = 2
    DATA = 3

class InteractiveAgent:
    """
    Author: Ondrej Lukas, ondrej.lukas@aic.cvut.cz
    This agent allows to manually play the Network Security Game

    """

    def __init__(self)->None:
        pass

    def move(self, observation: Observation)->Action:
        """
        Perform a move of the agent by
        - Selecting the action
        - Selecting the parameters
        - Returning the actions with parameters
        - Returns False if no action could be selected
        """
        # Get Action type to play
        action_type = get_action_type_from_stdin()
        if action_type:
            #get parameters of actions
            params = get_action_params_from_stdin(action_type, observation['state'])
            if params:
                action = Action(action_type, params)
                #print(f"Playing {action}")
                return action
        print(colored("Incorrect input, terminating the game!", "red"))
        # If something failed, avoid doing the move
        return False


def get_action_type_from_stdin()->ActionType:
    """
    Small function to call the function that does the selection of actions
    Probably not needed separatedly
    """
    print("Available Actions:")
    action_type = get_selection_from_user(ActionType, f"Select an action to play [0-{len(ActionType)-1}]: ")
    return action_type


def sanitize_user_input(input_string: str, action_type: ActionType, input_type: InputType):
    stripped = input_string.strip()
    if len(stripped) > 0:
        match action_type:
            case ActionType.ScanNetwork:
                if input_type == InputType.NETWORK:
                    splitted = stripped.split("/")
                    if len(splitted) == 2:
                        return splitted
                    else:
                        return False
                else:
                    splitted = stripped.split(" ")
                if len(splitted) == 1:
                    return splitted[0]
                else:
                    return False    
            case _:
                splitted = stripped.split(" ")
                if len(splitted) == 1:
                    return splitted[0]
                else:
                    return False
    else:
        return False


def get_action_params_from_stdin(action_type: ActionType, current: GameState)->dict:
    """
    Method which promts user to give parameters for given action_type
    """
    params = {}
    match action_type:
        case ActionType.ScanNetwork:
            user_input_host_src = input(f"Provide SOURCE host for selected action {action_type}: ")
            valid_input_src_host = sanitize_user_input(user_input_host_src, action_type, InputType.HOST)
            while not valid_input_src_host:
                print(colored("Incorrect input, desired format of host: X.X.X.X", "red"))
                user_input_host_src = input(f"Provide SOURCE host for selected action {action_type}: ")
                valid_input_src_host = sanitize_user_input(user_input_host_src, action_type, InputType.HOST)
            src_host = IP(valid_input_src_host)

            user_input = input(f"Provide network for selected action {action_type}: ")
            valid_input = sanitize_user_input(user_input, action_type, InputType.NETWORK)
            while not valid_input:
                print(colored("Incorrect input, desired format of network: X.X.X.X/mask", "red"))
                user_input = input(f"Provide network for selected action {action_type}: ")
                valid_input = sanitize_user_input(user_input, action_type, InputType.NETWORK)

            params = {"target_network": Network(valid_input[0], valid_input[1]), "source_host": src_host}
        
        case ActionType.ExploitService:
            user_input_host_src = input(f"Provide SOURCE host for selected action {action_type}: ")
            valid_input_src_host = sanitize_user_input(user_input_host_src, action_type, InputType.HOST)
            while not valid_input_src_host:
                print(colored("Incorrect input, desired format of host: X.X.X.X", "red"))
                user_input_host_src = input(f"Provide SOURCE host for selected action {action_type}: ")
                valid_input_src_host = sanitize_user_input(user_input_host_src, action_type, InputType.HOST)
            src_host = IP(valid_input_src_host)

            target_host = input(f"Provide TARGET host for selected action {action_type}: ")
            valid_input = sanitize_user_input(target_host, action_type, InputType.HOST)
            while not valid_input:
                print(colored("Incorrect input, desired format of host: X.X.X.X", "red"))
                target_host = input(f"Provide TARGET host for selected action {action_type}: ")
                valid_input = sanitize_user_input(target_host, action_type, InputType.HOST)
            trg_host = IP(valid_input)
            if str(trg_host) in current['known_services']:
                print(f"Known services in {trg_host}")
                service_dict = get_selection_from_user(current['known_services'][str(trg_host)], f"Select service to exploint [0-{len(current['known_services'][str(trg_host)])-1})]: ")
                service = Service(type=service_dict['type'], name=service_dict['name'], version=service_dict['version'], is_local=service_dict['is_local'])
                params = {"target_host": trg_host, "target_service": service, "source_host": src_host}
        
        case ActionType.ExfiltrateData:
            user_input_host_src = input(f"Provide SOURCE host for selected action {action_type}: ")
            valid_input_src_host = sanitize_user_input(user_input_host_src, action_type, InputType.HOST)
            while not valid_input_src_host:
                print(colored("Incorrect input, desired format of host: X.X.X.X", "red"))
                user_input_host_src = input(f"Provide SOURCE host for selected action {action_type}: ")
                valid_input_src_host = sanitize_user_input(user_input_host_src, action_type, InputType.HOST)
            src_host = IP(valid_input_src_host)

            if str(src_host) in current['known_data']:
                print(f"Known data in {src_host}")
                data = get_selection_from_user(current['known_data'][str(src_host)], f"Select data to exflitrate [0-{len(current['known_data'][str(src_host)])-1}]: ")
                if data:
                    user_input_host_trg = input(f"Provide TARGET host for data exfiltration: ")
                    valid_input_trg_host = sanitize_user_input(user_input_host_trg, action_type, InputType.HOST)
                    while not valid_input_trg_host:
                        print(colored("Incorrect input, desired format of host: X.X.X.X", "red"))
                        user_input_host_trg = input(f"Provide TARGET host for data exfiltration: ")
                        valid_input_trg_host = sanitize_user_input(user_input_host_trg, action_type, InputType.HOST)
                    trg_host = IP(valid_input_trg_host)
                    data_obj = Data(owner=data['owner'], id=data['id'])
                    params = {"target_host": trg_host, "data":data_obj, "source_host":src_host}
            else:
                print(f"Host {src_host} does not have any data yet.")
        case _:
            user_input_host_src = input(f"Provide SOURCE host for selected action {action_type}: ")
            valid_input_src_host = sanitize_user_input(user_input_host_src, action_type, InputType.HOST)
            while not valid_input_src_host:
                print(colored("Incorrect input, desired format of host: X.X.X.X", "red"))
                user_input_host_src = input(f"Provide SOURCE host for selected action {action_type}: ")
                valid_input_src_host = sanitize_user_input(user_input_host_src, action_type, InputType.HOST)
            src_host = IP(valid_input_src_host)

            user_input = input(f"Provide TARGET host for selected action {action_type}: ")
            valid_input = sanitize_user_input(user_input, action_type, InputType.HOST)
            while not valid_input:
                print(colored("Incorrect input, desired format of host: X.X.X.X", "red"))
                user_input = input(f"Provide TARGET host for selected action {action_type}: ")
                valid_input = sanitize_user_input(user_input, action_type, InputType.HOST)
            params = {"target_host": IP(valid_input), "source_host":src_host}
    return params


def get_selection_from_user(actiontypes: ActionType, prompt) -> ActionType:
    """
    Receive an ActionType object that contains all the options of actions
    Get the selection of action in text from the user in the stdin
    """
    option_dict = dict(enumerate(actiontypes))
    input_alive = True
    selected_option = None
    for index, option in option_dict.items():
        print(f"\t{index} - {option}")
    while input_alive:
        user_input = input(prompt)
        if user_input.lower() == "exit":
            input_alive = False
        else:
            try:
                selected_idx = int(user_input)
                selected_option = option_dict[selected_idx]
                input_alive = False
            except (ValueError, KeyError):
                print(colored(f"Please insert a number in range {min(option_dict.keys())}-{max(option_dict.keys())}!", 'red'))
    return selected_option

def print_current_state(new_state: GameState, reward: int = None, new_previous_state={}):
    """
    Prints GameState to stdout in formatted way
    """
    # Convert the dictionary parts of the state into a GameState obj
    state = GameState(new_state['controlled_hosts'], new_state['known_hosts'],new_state["known_services"], new_state["known_data"], new_state['known_networks'])
    previous_state = GameState(new_previous_state['controlled_hosts'], new_previous_state['known_hosts'],new_previous_state["known_services"], new_previous_state["known_data"], new_previous_state['known_networks'])

    def print_known_services(known_services, previous_state):
        if len(known_services) == 0:
            print(f"| {colored('SERVICES',None,attrs=['bold'])}: N/A")
        else:
            first = True
            services = {}
            if previous_state:
                for k in sorted(known_services.keys()):
                    if k in previous_state.known_services:
                        services[k] = [str(s) for s in sorted(known_services[k], key=lambda x: x['name'])]
                    else:
                        services[colored(k, 'yellow')] = [colored(str(s),'yellow') for s in sorted(known_services[k], key=lambda x: x['name'])]
            else:
                for k in sorted(known_services.keys()):
                    services[colored(k, 'yellow')] = [colored(str(s),'yellow') for s in sorted(known_services[k], key=lambda x: x['name'])]

            for host, service_list in services.items():
                if first:
                    print(f"| {colored('SERVICES',None,attrs=['bold'])}: {host}:")
                    for service in service_list:
                        print(f"|\t\t{service}")
                    first = False
                else:
                    print(f"|           {host}:")
                    for service in service_list:
                        print(f"|\t\t{service}")

    def print_known_data(known_data, previous_state):
        if len(known_data) == 0:
            print(f"| {colored('DATA',None,attrs=['bold'])}: N/A")
        else:
            first = True
            data = {}
            if previous_state:
                for k in sorted(known_data.keys()):
                    if k in previous_state.known_data:
                        data[k] = [str(d) for d in sorted(known_data[k], key=lambda x: x['owner'])]
                    else:
                        data[colored(k, 'yellow')] = [colored(str(d),'yellow') for d in sorted(known_data[k], key=lambda x: x['owner'])]
            else:
                for k in sorted(known_data.keys()):
                    data[colored(k, 'yellow')] = [colored(str(d),'yellow') for d in sorted(known_data[k], key=lambda x: x['owner'])]
            
            for host, data_list in data.items():
                if first:
                    print(f"| {colored('DATA',None,attrs=['bold'])}: {host}:")
                    for datapoint in data_list:
                        print(f"|\t\t{datapoint}")
                    first = False
                else:
                    print(f"|       {host}:")
                    for datapoint in data_list:
                        print(f"|\t\t{datapoint}")

    print(f"\n+============================================= {colored('CURRENT STATE','light_blue',attrs=['bold'])} (reward={reward}) ===============================================")
    if previous_state:
        logger.info(f'Prev state: {previous_state}')
        previous_nets =[]
        new_nets = []
        for net in state.known_networks:
            logger.info(f'net: {net}')
            logger.info(f'known_nets: {state.known_networks}')
            if net in previous_state.known_networks:
                previous_nets.append(net)
            else:
                new_nets.append(net)
        nets = [str(str(n['ip']) + '/' + str(n['mask'])) for n in sorted(previous_nets, key=lambda x: x['ip'])] + [colored(str(n['ip']) + '/' + str(n['mask']), 'yellow') for n in sorted(new_nets, key=lambda x: x['ip'])]
    else:
        nets = [colored(str(net['ip']+'/'+net['mask']), 'yellow') for net in sorted(state.known_networks, key=lambda x: x['ip'])]
    print(f"| {colored('NETWORKS',None,attrs=['bold'])}: {', '.join(nets)}")
    print("+----------------------------------------------------------------------------------------------------------------------")
    if previous_state:
        previous_hosts =[]
        new_hosts = []
        for host in state.known_hosts:
            if host in previous_state.known_hosts:
                previous_hosts.append(host)
            else:
                new_hosts.append(host)
        hosts = [str(h['ip']) for h in sorted(previous_hosts, key=lambda x: x['ip'])] + [colored(str(h['ip']), 'yellow') for h in sorted(new_hosts, key=lambda x: x['ip'])]
    else:
        hosts = [colored(str(host['ip']), 'yellow') for host in sorted(state.known_hosts, key=lambda x: x['ip'])]
    print(f"| {colored('KNOWN_H',None,attrs=['bold'])}: {', '.join(hosts)}")
    print("+----------------------------------------------------------------------------------------------------------------------")
    if previous_state:
        previous_hosts =[]
        new_hosts = []
        for host in state.controlled_hosts:
            if host in previous_state.controlled_hosts:
                previous_hosts.append(host)
            else:
                new_hosts.append(host)
        owned_hosts = [str(h['ip']) for h in sorted(previous_hosts, key=lambda x: x['ip'])] + [colored(str(h['ip']), 'yellow') for h in sorted(new_hosts, key=lambda x: x['ip'])]
    else:
        owned_hosts = [colored(str(host['ip']), 'yellow') for host in sorted(state.controlled_hosts, key=lambda x: x['ip'])]    
    print(f"| {colored('OWNED_H',None,attrs=['bold'])}: {', '.join(owned_hosts)}")
    print("+----------------------------------------------------------------------------------------------------------------------")
    print_known_services(state.known_services, previous_state)
    print("+----------------------------------------------------------------------------------------------------------------------")
    print_known_data(state.known_data, previous_state)
    print("+======================================================================================================================\n")

def receive_data(client_socket):
    """
    Receive data from server
    """
    # Receive data from the server
    data = client_socket.recv(8192).decode()
    logger.info(f"Data received from env: {data}")
    data_dict = json.loads(data)
    return data_dict

def register(client_socket):
    """
    Register in the game with a nickname
    """
    try:
        status, observation, message = step(client_socket, '')

        if 'Insert your nick' in message:
            # Send the nick
            print(colored(message, 'light_cyan'))
            player_name = input(colored("Answer: ", 'light_cyan'))
        
            while not player_name or len(player_name)==0:
                print(colored("Incorrect input.", "red"))
                player_name = input(colored("Answer: ", 'light_cyan'))
            
            message_dict = {'PutNick': player_name}
            message_str = json.dumps(message_dict)
            
            status, observation, message = step(client_socket, message_str)
            return status, observation, message
        else:
            return False, False, False
    except Exception as e:
        logger.error(f'Exception in register(): {e}')

def choose_side(client_socket, message):
    """
    Choose your side
    """
    try:

        if 'Which side are' in message:
            # Choose side
            print(colored(message, 'light_cyan'))
            side = input(colored("Answer: ", 'light_cyan'))
            while not side or len(side)==0:
                print(colored("Incorrect input.", "red"))
                side = input(colored("Answer: ", 'light_cyan'))
            message_dict = {'ChooseSide': side}
            message_str = json.dumps(message_dict)
            
            status, observation, message = step(client_socket, message_str)
            return status, observation, message
        else:
            return {}, {}, ''
    except Exception as e:
        print(e)
        logger.error(f'Exception in choose_side(): {e}')

def step(client_socket, action):
    """
    Send an action and receive a response
    """
    try:
        logger.info(f'Doing step')
        logger.info(f'Sending: {action}')

        if type(action) == Action:
            action = action.as_json()

        client_socket.sendall(action.encode())
        data_dict = receive_data(client_socket)
        logger.info(f'Received: {data_dict}')
        try:
            status = data_dict['status']
        except KeyError:
            status = {}
        try:
            observation = data_dict['observation']
        except KeyError:
            observation = {}
        try:
            message = data_dict['message']
        except KeyError:
            message = ''
        return status, observation, message
    except Exception as e:
        logger.error(f'Exception in step(): {e}')

def play(host, port, agent, num_episodes=None):
    """
    Play the game
    """
    try:
        # Open connection to the game server
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Connect to the server
        client_socket.connect((host, port))

        # Register
        status, observation_dict, message = register(client_socket)

        if status:
            # Choose side
            status, observation_dict, message = choose_side(client_socket, message)
            # This status, observation and message are the ones that start the game
        else:
            logger.info(f'Could not register in the server.')
            return False
        
        if 'That side does not exists.' in message:
            logger.info(message)
            print(message)
            sys.exit(-1)
        
        if not status:
            return False

        episode_counter = 0
        stop = False

        while not stop and observation_dict:
            print(colored(message, 'light_cyan'))
            
            # Convert the state in observation from json string to dict
            logger.info(f'Interpreted: \n\tObservation:{observation_dict}')

            # The first 'empty' previous state should be a dictionary game state, but empty. But represented as a dictionary. This can only be done from the json representation
            empty_game_state = json.loads(GameState(known_networks=[], known_hosts=[], controlled_hosts=[], known_services={}, known_data={}).as_json())
            # Must be a dict
            previous_observation = {'state': empty_game_state, 'reward': 0, 'end': False, 'info': {}}

            while not observation_dict['end'] and not stop:
                print_current_state(observation_dict['state'], observation_dict['reward'], previous_observation['state'])
                action = agent.move(observation_dict)
                if action:
                    status, observation_dict, message = step(client_socket, action)
                previous_observation = observation_dict
            episode_counter +=1
            print(colored(f"\nEpisode over! Reason {observation_dict['info']}", 'light_cyan'))
            if input(colored("\nDo you want to play again? Y/n: ", 'light_cyan')) in ["Y", 'y']:
                print("\n################################################ STARTING NEW EPISODE ################################################\n")
            else:
                stop = True
    except KeyboardInterrupt:
        logger.info('Agent pressed CTRL-C')
        print('\nAgent pressed CTRL-C')
        client_socket.close()
    finally:
        # Close the socket
        client_socket.close()

def main() -> None:
    """
    Function to run the run the interactive agent
    """
    parser = argparse.ArgumentParser()
    #parser.add_argument("--task_config_file", help="Reads the task definition from a configuration file", default=path.join(path.dirname(__file__), 'netsecenv-task.yaml'), action='store', required=False)
    parser.add_argument("--rb_log_directory", help="directory to store the logs", default="env/logs/replays", action='store', required=False)
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    
    args = parser.parse_args()
    
    # Make sure the folder for replays exists
    if not os.path.exists(args.rb_log_directory):
        os.makedirs(args.rb_log_directory)

    logger.info('Creating the agent')
    agent = InteractiveAgent()
    play(args.host, args.port, agent, None)

if __name__ == '__main__':
    main()