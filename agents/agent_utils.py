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
                    # Do not consider local services, which are internal to the host
                    if not service.is_local:
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
    #logger.info(f'IPS-CONCEPT: Received obs with CH: {state.controlled_hosts}, KH: {state.known_hosts}, KS: {state.known_services}, KD: {state.known_data}, NETs:{state.known_networks}')
    # state.controlled_hosts: set
    # state.known_hosts: set
    # state.known_networks: set
    # state.known_data: dict. {'ip': Data object}
    #   Data - Object
    #       data.ownwer
    #       data.id
    # state.known_services: dict. {'ip': Service object}
    #   Service - Object
    #       service.name 'openssh' (what in the configuration is 'type'). The service name is derived from the nmap services https://svn.nmap.org/nmap/nmap-services
    #       service.type 'passive' | 'active' (This is added by our env when finding the )
    #       service.version : '1.1.1.'
    #       service.is_local: Bool
        

    new_observation = None
    state = observation.state
    reward = observation.reward
    end = observation.end
    info = observation.info

    # Here we keep the mapping of concepts to values
    concept_mapping = {'controlled_hosts': {}, 'known_hosts': {}, 'known_services': {}, 'known_data': {}, 'known_networks': {}}

    # Hosts

    # Host are separated according to their services. So we only do the separation if they have services, if not, just all together.
    # The concept of type of hosts comes from the type of data they may have inside
    db_hosts_words = ['sql', 'dbase', 'mongo', 'redis', 'database']
    db_hosts = set()
    db_hosts_idx = IP('db')
    web_hosts_words = ['http', 'web']
    web_hosts = set()
    web_hosts_idx = IP('web')
    remote_hosts_words = ['ssh', 'telnet', 'ms-wbt-server', 'remote', 'shell']
    remote_hosts = set()
    remote_hosts_idx = IP('remote')
    files_hosts_words = ['microsoft-ds', 'nfs', 'ftp']
    files_hosts = set()
    files_hosts_idx = IP('files')
    external_hosts = set()
    external_hosts_idx = IP('external')
    unknown_hosts = set()
    unknown_hosts_idx = IP('unknown')

    # To have a reverse dict of ips to concepts so we can assign the controlled hosts fast from the known hosts
    # This is a performance trick
    ip_to_concept = {}


    # Convert the known hosts
    logger.info(f'\tI2C: state known hosts: {state.known_hosts}')
    logger.info(f'\tI2C: state controlled hosts: {state.controlled_hosts}')

    logger.info(f'\tI2C: Converting')
    for host in state.known_hosts:
        # Is it external
        if not host.is_private():
            external_hosts.add(host)
            concept_mapping['known_hosts'][external_hosts_idx] = external_hosts
            ip_to_concept[host] = external_hosts_idx
            continue

        # Does it have services?
        elif host in list(state.known_services.keys()):
            for service in state.known_services[host]:
                # First, all hosts with services are 'unknonw'. It is faster to add it to unknown and then assign a new one if necessary
                unknown_hosts.add(host)
                # db
                for word in db_hosts_words:
                    if word in service.name:
                        db_hosts.add(host)
                        concept_mapping['known_hosts'][db_hosts_idx] = db_hosts
                        ip_to_concept[host] = db_hosts_idx
                        unknown_hosts.discard(host)
                        break # one word is enough
                # web
                for word in web_hosts_words:
                    if word in service.name:
                        web_hosts.add(host)
                        concept_mapping['known_hosts'][web_hosts_idx] = web_hosts
                        ip_to_concept[host] = web_hosts_idx
                        unknown_hosts.discard(host)
                        break # one word is enough
                # remote
                for word in remote_hosts_words:
                    if word in service.name:
                        remote_hosts.add(host)
                        concept_mapping['known_hosts'][remote_hosts_idx] = remote_hosts
                        ip_to_concept[host] = remote_hosts_idx
                        unknown_hosts.discard(host)
                        break # one word is enough
                # files hosts 
                for word in files_hosts_words:
                    if word in service.name:
                        files_hosts.add(host)
                        concept_mapping['known_hosts'][files_hosts_idx] = files_hosts
                        ip_to_concept[host] = files_hosts_idx
                        unknown_hosts.discard(host)
                        break # one word is enough
        else:
            # Host does not have any service yet
            unknown_hosts.add(host)

        concept_mapping['known_hosts'][unknown_hosts_idx] = unknown_hosts
        ip_to_concept[host] = unknown_hosts_idx
    
    # Use the ip to concept to gather the controlled hosts concepts
    for ip, object in ip_to_concept.items():
        concept_mapping['controlled_hosts'][object] = concept_mapping['known_hosts'][object]

    logger.info(f'\tI2C: New concept known_hosts: {concept_mapping['known_hosts']}')
    logger.info(f'\tI2C: New concept controlled_hosts: {concept_mapping['controlled_hosts']}')

    # Convert Networks

    # The set for my networks. Networks here you control a host
    my_networks = set()
    # The index object
    my_nets = Network('mynet', 24)

    # The set
    unknown_networks = set()
    # The index object
    unknown_nets = Network('unknown', 24)

    #logger.info(f'\tI2C: state known networks: {state.known_networks}')
    for network in state.known_networks:
        # net_assigned is only to speed up the process when a network has been added because the agent
        # controlls a host there, or two.
        net_assigned = False
        #logger.info(f'\tI2C: Trying to convert network {network}. NetIP:{network.ip}. NetMask: {network.mask}')

        # Find the mynet networks
        for controlled_ip in state.controlled_hosts:
            #logger.info(f'\t\tI2C: Checking with ip {controlled_ip}')
            if ipaddress.IPv4Address(controlled_ip.ip) in ipaddress.IPv4Network(f'{network.ip}/{network.mask}'):
                #logger.info(f'\t\tI2C: Controlled IP {controlled_ip} is in network {network}. So mynet.')
                my_networks.add(network)
                net_assigned = True
                # Store mynets
                concept_mapping['known_networks'][my_nets] = my_networks
                break
        # If this network was assigned to mynet, dont try to assign it again
        if net_assigned:
            continue

        # Find if we know hosts in this network, if we do, assign new name
        # and remove from unknown
        number_hosts = 0
        for known_host in state.known_hosts:
            if ipaddress.IPv4Address(known_host.ip) in ipaddress.IPv4Network(f'{network.ip}/{network.mask}'):
                # There are hosts we know in this network
                number_hosts += 1
        if number_hosts:
            # The index
            new_net = Network('net' + str(number_hosts), 24)
            try:
                # Did we have any?
                net_with_hosts = concept_mapping['known_networks'][new_net]
            except KeyError:
                net_with_hosts = set()

            net_with_hosts.add(network)
            concept_mapping['known_networks'][new_net] = net_with_hosts
            # Remove from unknowns
            try:
                unknowns = concept_mapping['known_networks']['unknown/24']
                unknowns.discard(network)
            except KeyError:
                pass
            # Continue with next net
            continue

        # Still we didnt assigned this net, so unknown
        #logger.info(f'\t\tI2C: It was not my net, so unknown: {network}')
        unknown_networks.add(network)
        # Store unknown nets
        concept_mapping['known_networks'][unknown_nets] = unknown_networks
        # In the future we can lost a controlling host in  a net, so if we add it to unknown, delete from other groups

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
        # parameters = {"target_host": IP(parameters_dict["target_host"]["ip"]), "target_service": Service(parameters_dict["target_service"]["name"], parameters_dict["target_service"]["type"], parameters_dict["target_service"]["version"], parameters_dict["target_service"]["is_local"]), "source_host": IP(parameters_dict["source_host"]["ip"])}
        pass
    elif action._type == ActionType.ExfiltrateData:
        # parameters = {"target_host": IP(parameters_dict["target_host"]["ip"]), "source_host": IP(parameters_dict["source_host"]["ip"]), "data": Data(parameters_dict["data"]["owner"],parameters_dict["data"]["id"])}
        pass
    elif action._type == ActionType.FindData:
        # parameters = {"source_host": IP(parameters_dict["source_host"]["ip"]), "target_host": IP(parameters_dict["target_host"]["ip"])}
        pass
    elif action._type == ActionType.ScanNetwork:
        target_net = action.parameters['target_network']
        for net in concept_mapping['known_networks']:
            if target_net == net:
                new_target_network = random.choice(list(concept_mapping['known_networks'][net]))
                break
        # Change the src host from concept to ip
        #new_src_host = action.parameters['source_host']
        new_src_host = concept_mapping['known_hosts'][action.parameters['source_host']]
        action = Action(ActionType.ScanNetwork, params={"source_host": new_src_host, "target_network": new_target_network} )
    elif action._type == ActionType.FindServices:
        # parameters = {"source_host": IP(parameters_dict["source_host"]["ip"]), "target_host": IP(parameters_dict["target_host"]["ip"])}
        pass

    return action