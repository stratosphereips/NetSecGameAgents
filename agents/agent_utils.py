"""
Collection of functions which are intended for agents to share.
Functionality which is specific for a certain type of agent should be implementented
directly in the agent class. 

author: Ondrej Lukas - ondrej.lukas@aic.fel.cvut.cz
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))
#with the path fixed, we can import now
from env.game_components import Action, ActionType, GameState, Observation

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
    out: observation with concepts
    """
    new_observation = None
    state = observation.state
    reward = observation.reward
    end = observation.end
    info = observation.info

    # state.controlled_hosts
    # state.known_hosts
    # state.known_services
    # state.known_data
    # state.known_networks
    #logger.info(f'IPS-CONCEPT: Received obs with CH: {state.controlled_hosts}, KH: {state.known_hosts}, KS: {state.known_services}, KD: {state.known_data}, NETs:{state.known_networks}')

    new_observation = Observation(state, reward, end, info)
    return new_observation

def convert_concepts_to_ips(action, logger):
    """
    Function to convert the concepts learned before into IPs and networks
    so the env knows where to really act
    in: action for concepts
    out: action for IPs
    """
    #logger.info(f'CONCEPT-ACTION: Received action with ')
    return action