"""
Collection of functions which are intended for agents to share.
Functionality which is specific for a certain type of agent should be implementented
directly in the agent class. 

author: Ondrej Lukas - ondrej.lukas@aic.fel.cvut.cz
"""
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))
#with the path fixed, we can import now
from env.game_components import Action, ActionType, GameState, Observation, Data, IP, Network
import ipaddress

def generate_valid_actions(state: GameState)->list:
    """Function that generates a list of all valid actions in a given state"""
    valid_actions = set()
    for src_host in state.controlled_hosts:
        # Network Scans
        for network in state.known_networks:
            # TODO ADD neighbouring networks
            # Only scan local networks from local hosts
            if network.is_private() and src_host.is_private():
                valid_actions.add(Action(ActionType.ScanNetwork, params={"target_network": network, "source_host": src_host,}))
        # Service Scans
        for host in state.known_hosts:
            # Do not try to scan a service from hosts outside local networks towards local networks
            if host.is_private() and src_host.is_private():
                valid_actions.add(Action(ActionType.FindServices, params={"target_host": host, "source_host": src_host,}))
        # Service Exploits
        for host, service_list in state.known_services.items():
            # Only exploit local services from local hosts
            if host.is_private() and src_host.is_private():
                for service in service_list:
                    valid_actions.add(Action(ActionType.ExploitService, params={"target_host": host,"target_service": service,"source_host": src_host,}))
    # Data Scans
    for host in state.controlled_hosts:
        valid_actions.add(Action(ActionType.FindData, params={"target_host": host, "source_host": host}))

    # Data Exfiltration
    for src_host, data_list in state.known_data.items():
        for data in data_list:
            for trg_host in state.controlled_hosts:
                if trg_host != src_host:
                    valid_actions.add(Action(ActionType.ExfiltrateData, params={"target_host": trg_host, "source_host": src_host, "data": data}))
    return list(valid_actions)

def state_as_ordered_string(state:GameState)->str:
    """Function for generating string representation of a SORTED gamestate components. Can be used as key for dictionaries."""
    ret = ""
    ret += f"nets:[{','.join([str(x) for x in sorted(state.known_networks)])}],"
    ret += f"hosts:[{','.join([str(x) for x in sorted(state.known_hosts)])}],"
    ret += f"controlled:[{','.join([str(x) for x in sorted(state.controlled_hosts)])}],"
    ret += "services:{"
    for host in sorted(state.known_services.keys()):
        ret += f"{host}:[{','.join([str(x) for x in sorted(state.known_services[host])])}]"
    ret += "},data:{"
    for host in sorted(state.known_data.keys()):
        ret += f"{host}:[{','.join([str(x) for x in sorted(state.known_data[host])])}]"
    ret += "}"
    return ret   

def recompute_reward(self, observation: Observation) -> Observation:
    """
    Redefine how an agent recomputes the inner reward
    in: Observation object
    out: Observation object
    """
    new_observation = None
    state = observation['state']
    reward = observation['reward']
    end = observation['end']
    info = observation['info']

    # The rewards hare are the originals from the env. 
    # Each agent can do this differently
    if info and info['end_reason'] == 'detected':
        # Reward when we are detected
        reward = -100
    elif info and info['end_reason'] == 'goal_reached':
        # Reward when we win
        reward = 100
    elif info and info['end_reason'] == 'max_steps':
        # Reward when we hit max steps
        reward = -1
    else:
        # Reward when we hit max steps
        reward = -1
    
    new_observation = Observation(GameState.from_dict(state), reward, end, info)
    return new_observation


def convert_ips_to_concepts(observation, logger):
    """
    Function to convert the IPs and networks in the observation into a concept 
    so the agent is not dependent on IPs and specific values
    in: observation with IPs
    out: observation with concepts, dict with concept_mapping
    """
    new_observation = None
    state = observation.state
    reward = observation.reward
    end = observation.end
    info = observation.info

    # Here we keep the mapping of concepts to values
    concept_mapping = {'controlled_hosts': {}, 'known_hosts': {}, 'known_services': {}, 'known_data': {}, 'known_networks': {}}

    # {192.168.1.2: {Data(owner='User2', id='Data2FromServer1'), Data(owner='User1', id='DataFromServer1'), Data(owner='User1', id='Data3FromServer1')}, 213.47.23.195: {Data(owner='User1', id='DataFromServer1')}}

    # state.controlled_hosts
    # state.known_hosts
    # state.known_services
    # state.known_data
    # state.known_networks
    #logger.info(f'IPS-CONCEPT: Received obs with CH: {state.controlled_hosts}, KH: {state.known_hosts}, KS: {state.known_services}, KD: {state.known_data}, NETs:{state.known_networks}')

    state_networks = {}
    my_networks = []
    unknown_networks = []
    #logger.info(f'\tI2C: state known networks: {state.known_networks}')
    for network in state.known_networks:
        net_assigned = False
        #logger.info(f'\tI2C: Trying to convert network {network}. NetIP:{network.ip}. NetMask: {network.mask}')
        for controlled_ip in state.controlled_hosts:
            #logger.info(f'\t\tI2C: Checking with ip {controlled_ip}')
            if ipaddress.IPv4Address(controlled_ip.ip) in ipaddress.IPv4Network(f'{network.ip}/{network.mask}'):
                #logger.info(f'\t\tI2C: Controlled IP {controlled_ip} is in network {network}. So mynet.')
                my_networks.append(network)
                net_assigned = True
                break
        if net_assigned:
            continue
        # Still we didnt assigned this net, so unknown
        #logger.info(f'\t\tI2C: It was not my net, so unknown: {network}')
        unknown_networks.append(network)

    my_nets = Network('mynet', 24)
    unknown_nets = Network('unknown', 24)
    concept_mapping['known_networks'][my_nets] = my_networks
    concept_mapping['known_networks'][unknown_nets] = unknown_networks
    state_networks = {net for net in concept_mapping['known_networks']}

    new_state = GameState(state.controlled_hosts, state.known_hosts, state.known_services, state.known_data, state_networks)
    new_observation = Observation(new_state, reward, end, info)
    return new_observation, concept_mapping

def convert_concepts_to_actions(action, concept_mapping, logger):
    """
    Function to convert the concepts learned before into IPs and networks
    so the env knows where to really act
    in: action for concepts
    out: action for IPs
    """
    #logger.info(f'C2IP: Action to deal with: {action}')
    #logger.info(f'\tC2IP: Action type: {action._type}')
    if action._type == ActionType.ExploitService:
        pass
    elif action._type == ActionType.ExfiltrateData:
        pass
    elif action._type == ActionType.FindData:
        pass
    elif action._type == ActionType.ScanNetwork:
        #known_networks = concept_mapping['known_networks']
        target_net = action.parameters['target_network']
        for net in concept_mapping['known_networks']:
            if target_net == net:
                new_target_network = random.choice(concept_mapping['known_networks'][net])
                break
        new_src_host = action.parameters['source_host']
        action = Action(ActionType.ScanNetwork, params={"source_host": new_src_host, "target_network": new_target_network} )
    elif action._type == ActionType.FindServices:
        pass

    return action