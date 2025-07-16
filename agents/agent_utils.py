"""
Collection of functions which are intended for agents to share.
Functionality which is specific for a certain type of agent should be implementented
directly in the agent class. 

author: Ondrej Lukas - ondrej.lukas@aic.fel.cvut.cz
"""
import random
import ipaddress
from AIDojoCoordinator.game_components import Action, ActionType, GameState, Observation, IP, Network

def estimate_subnetwork_from_ip(ip:IP)-> Network:
    """
    Estimate the subnetwork of a given IP address.
    Returns a Network object with the IP and a default mask of 24.
    """
    octets = str(ip).split('.')
    if str(ip).startswith("192.168."):
        return Network(f"{octets[0]}.{octets[1]}.{octets[2]}.0",24)
    elif str(ip).startswith("10."):
        return Network(f"{octets[0]}.0.0.0", 8)
    elif str(ip).startswith("172."):
        return Network(f"{octets[0]}.{octets[1]}.0.0", 16)
    else:
        return Network(f"{octets[0]}.{octets[1]}.{octets[2]}.0", 24)  # Fallback option, assuming a /24 subnet

def estimate_neighboring_networks(network: Network, offset:int=1) -> set:
    """
    Estimate the neighboring networks of a given network.
        - If the input network is private, only return private neighbors.
        - If the input network is public, return all neighbors.
    Returns a set of Network objects that are considered neighboring networks.
    """
    neighboring_networks = set()
    
    # Convert the network to an IP network object
    ip_network = ipaddress.ip_network(f"{network.ip}/{network.mask}", strict=False)
    size = ip_network.num_addresses
    base = int(ip_network.network_address)
    is_private = ip_network.is_private
    # Generate neighboring networks by offsetting the base address
    # We use the size of the network to determine the step size for neighbors
    # We skip the original network (offset 0) and only add neighbors within the specified offset range
    for i in range(-offset, offset + 1):
        if i == 0:
            continue  # skip original
        try:
            new_base_ip = ipaddress.IPv4Address(base + i * size)
            neighbor = ipaddress.ip_network(f"{new_base_ip}/{ip_network.prefixlen}")
            if is_private and neighbor.is_private:
                neighboring_networks.add(Network(str(new_base_ip), neighbor.prefixlen))
        except ValueError:
            # If the neighbor is not a valid network, we skip it
            continue
    return neighboring_networks

def heuristic_network_expansion(state: GameState, net_offset:int, explore_known_hosts:bool=True) -> GameState:
    """
    Expands the known network space by estimating neighboring networks and subnetworks from known hosts.
    """
    extended_networks = state.known_networks.copy()
    # estimate neighboring networks for each known network
    for network in state.known_networks:
        neighboring_networks = estimate_neighboring_networks(network, net_offset)
        extended_networks.update(neighboring_networks)
    if explore_known_hosts:
        # estimate subnetwork for each known host
        for host in state.known_hosts:
            subnetwork = estimate_subnetwork_from_ip(host)
            extended_networks.add(subnetwork)
    # update the state with the extended networks
    new_state = GameState(
        controlled_hosts=state.controlled_hosts,
        known_hosts=state.known_hosts,
        known_services=state.known_services,
        known_data=state.known_data,
        known_networks=extended_networks
    )
    return new_state

def generate_valid_actions_concepts(state: GameState)->list:
    """Function that generates a list of all valid actions in a given state"""
    valid_actions = set()
    for source_host in state.controlled_hosts:
        # Network Scans
        for network in state.known_networks:
            # TODO ADD neighbouring networks
            # Only scan local networks from local hosts
            if network.is_private() and source_host.is_private():
                valid_actions.add(Action(ActionType.ScanNetwork, parameters={"target_network": network, "source_host": source_host,}))
        # Service Scans
        for host in state.known_hosts:
            # Do not try to scan a service from hosts outside local networks towards local networks
            if host.is_private() and source_host.is_private():
                valid_actions.add(Action(ActionType.FindServices, parameters={"target_host": host, "source_host": source_host,}))
        # Service Exploits
        for host, service_list in state.known_services.items():
            # Only exploit local services from local hosts
            if host.is_private() and source_host.is_private():
                # Only exploits hosts we do not control
                if host not in state.controlled_hosts:
                    for service in service_list:
                        # Do not consider local services, which are internal to the host
                        if not service.is_local:
                            valid_actions.add(Action(ActionType.ExploitService, parameters={"target_host": host,"target_service": service,"source_host": source_host,}))
    # Find Data Scans
    for host in state.controlled_hosts:
        valid_actions.add(Action(ActionType.FindData, parameters={"target_host": host, "source_host": host}))

    # Data Exfiltration
    for source_host, data_list in state.known_data.items():
        for data in data_list:
            for target_host in state.controlled_hosts:
                if target_host != source_host:
                    valid_actions.add(Action(ActionType.ExfiltrateData, parameters={"target_host": target_host, "source_host": source_host, "data": data}))
    return list(valid_actions)

def generate_valid_actions(state: GameState, include_blocks=False)->list:
    """Function that generates a list of all valid actions in a given state"""
    valid_actions = set()
    def is_fw_blocked(state, src_ip, dst_ip)->bool:
        blocked = False
        try:
            blocked = dst_ip in state.known_blocks[src_ip]
        except KeyError:
            pass #this src ip has no known blocks
        return blocked 

    for src_host in state.controlled_hosts:
        #Network Scans
        for network in state.known_networks:
            # TODO ADD neighbouring networks
            valid_actions.add(Action(ActionType.ScanNetwork, parameters={"target_network": network, "source_host": src_host,}))
        # Service Scans
        for host in state.known_hosts:
            if not is_fw_blocked(state, src_host,host):
                valid_actions.add(Action(ActionType.FindServices, parameters={"target_host": host, "source_host": src_host,}))
        # Service Exploits
        for host, service_list in state.known_services.items():
            if not is_fw_blocked(state, src_host,host):
                for service in service_list:
                    valid_actions.add(Action(ActionType.ExploitService, parameters={"target_host": host,"target_service": service,"source_host": src_host,}))
    # Data Scans
    for host in state.controlled_hosts:
        if not is_fw_blocked(state, src_host,host):
            valid_actions.add(Action(ActionType.FindData, parameters={"target_host": host, "source_host": host}))

    # Data Exfiltration
    for src_host, data_list in state.known_data.items():
        for data in data_list:
            for trg_host in state.controlled_hosts:
                if trg_host != src_host:
                    if not is_fw_blocked(state, src_host,trg_host):
                        valid_actions.add(Action(ActionType.ExfiltrateData, parameters={"target_host": trg_host, "source_host": src_host, "data": data}))
    
    if include_blocks:
        # BlockIP
        if include_blocks:
            for src_host in state.controlled_hosts:
                for target_host in state.controlled_hosts:
                    if not is_fw_blocked(state, src_host,target_host):
                        for blocked_ip in state.known_hosts:
                            valid_actions.add(Action(ActionType.BlockIP, {"target_host":target_host, "source_host":src_host, "blocked_host":blocked_ip}))
    return list(valid_actions)    

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
    if info and info['end_reason'] == 'blocked':
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

    # Host are separated according to their services and function. So we only do the separation if they have services, if not, they are 'unknown'
    # The special case are the hosts we control in the local net, those are 'mineX' with X being a counter.
    # The concept of type of hosts comes from the type of data they may have inside or function
    db_hosts_words = ['sql', 'dbase', 'mongo', 'redis', 'database']
    #db_hosts = set()
    db_hosts_concept = 'db'
    web_hosts_words = ['http', 'web']
    #web_hosts = set()
    web_hosts_concept = 'web'
    remote_hosts_words = ['ssh', 'telnet', 'ms-wbt-server', 'remote', 'shell']
    #remote_hosts = set()
    remote_hosts_concept = 'remote'
    files_hosts_words = ['microsoft-ds', 'nfs', 'ftp']
    #files_hosts = set()
    files_hosts_concept = 'files'
    #external_hosts = set()
    external_hosts_concept = 'external'
    unknown_hosts = set()
    unknown_hosts_concept = 'unknown'
    #my_hosts = set()
    my_hosts_concept = 'mine'

    # To have a reverse dict of ips to concepts so we can assign the controlled hosts fast from the known hosts
    # This is a performance trick
    # The ip can have multiple concepts ip_to_concept['1.1.1.1'] = {'web', 'db'}
    ip_to_concept = {}


    # Convert controlled hosts    
    # The controlled hosts due to exploiting are not the 'unknown' concept anymore. Now they are concept 'controlled'
    """
    counter = 0
    for host in state.controlled_hosts:
        # Is it external?
        if not host.is_private():
            #external_hosts.add(host)
            external_hosts_idx = IP(external_hosts_concept+str(counter))
            counter += 1
            concept_mapping['controlled_hosts'][external_hosts_idx] = host
            concept_mapping['known_hosts'][external_hosts_idx] = host
            ip_to_concept[host] = {external_hosts_idx}
        else:
            # It is local
            my_hosts_idx = IP(my_hosts_concept+str(counter))
            counter += 1
            concept_mapping['controlled_hosts'][my_hosts_idx] = host
            concept_mapping['known_hosts'][my_hosts_idx] = host
            ip_to_concept[host] = {my_hosts_idx}
    """


    # Convert the known hosts

    logger.info(f'\tI2C: Real state known hosts: {state.known_hosts}')
    logger.info(f'\tI2C: Real state controlled hosts: {state.controlled_hosts}')

    #logger.info(f'\tI2C: Converting')
    # Counter to separate concepts in same category
    counter = 0
    for host in state.known_hosts:
        counter += 1
        # Is it external and it is not in ip_to_concept, so it is not controlled
        #if not host.is_private() and not ip_to_concept[host]:
        if not host.is_private():
            external_hosts_idx = IP(external_hosts_concept+str(counter))
            concept_mapping['known_hosts'][external_hosts_idx] = host
            try:
                ip_to_concept[host].add(external_hosts_idx)
            except KeyError:
                ip_to_concept[host] = {external_hosts_idx}
            # Is it also controlled?
            if host in state.controlled_hosts:
                concept_mapping['controlled_hosts'][external_hosts_idx] = host
            continue


        # Does it have services?
        elif host in list(state.known_services.keys()):
            for service in state.known_services[host]:
                # First, all hosts with services are 'unknonw'. It is faster to add it to unknown and then assign a new one if necessary
                unknown_hosts.add(host)

                # The same host can have multiple services, so it can be in multiple concepts

                # db
                for word in db_hosts_words:
                    if word in service.name:
                        #db_hosts.add(host)
                        db_hosts_idx = IP(db_hosts_concept+str(counter))
                        concept_mapping['known_hosts'][db_hosts_idx] = host
                        try:
                            ip_to_concept[host].add(db_hosts_idx)
                        except KeyError:
                            ip_to_concept[host] = {db_hosts_idx}
                        unknown_hosts.discard(host)
                        # Is it also controlled?
                        if host in state.controlled_hosts:
                            concept_mapping['controlled_hosts'][db_hosts_idx] = host
                        break # one word is enough
                # web
                for word in web_hosts_words:
                    if word in service.name:
                        #web_hosts.add(host)
                        web_hosts_idx = IP(web_hosts_concept+str(counter))
                        concept_mapping['known_hosts'][web_hosts_idx] = host
                        try:
                            ip_to_concept[host].add(web_hosts_idx)
                        except KeyError:
                            ip_to_concept[host] = {web_hosts_idx}
                        unknown_hosts.discard(host)
                        # Is it also controlled?
                        if host in state.controlled_hosts:
                            concept_mapping['controlled_hosts'][web_hosts_idx] = host
                        break # one word is enough
                # remote
                for word in remote_hosts_words:
                    if word in service.name:
                        #remote_hosts.add(host)
                        remote_hosts_idx = IP(remote_hosts_concept+str(counter))
                        concept_mapping['known_hosts'][remote_hosts_idx] = host
                        try:
                            ip_to_concept[host].add(remote_hosts_idx)
                        except KeyError:
                            ip_to_concept[host] = {remote_hosts_idx}
                        unknown_hosts.discard(host)
                        # Is it also controlled?
                        if host in state.controlled_hosts:
                            concept_mapping['controlled_hosts'][remote_hosts_idx] = host
                        break # one word is enough
                # files hosts 
                for word in files_hosts_words:
                    if word in service.name:
                        #files_hosts.add(host)
                        files_hosts_idx = IP(files_hosts_concept+str(counter))
                        concept_mapping['known_hosts'][files_hosts_idx] = host
                        try:
                            ip_to_concept[host].add(files_hosts_idx)
                        except KeyError:
                            ip_to_concept[host] = {files_hosts_idx}
                        unknown_hosts.discard(host)
                        # Is it also controlled?
                        if host in state.controlled_hosts:
                            concept_mapping['controlled_hosts'][files_hosts_idx] = host
                        break # one word is enough
        else:
            # These are all the devices without services
            # A device can be controlled (specially in the first state from the env)
            if host in state.controlled_hosts:
                # Yes it is controlled
                my_hosts_idx = IP(my_hosts_concept+str(counter))
                concept_mapping['controlled_hosts'][my_hosts_idx] = host
                concept_mapping['known_hosts'][my_hosts_idx] = host
                ip_to_concept[host] = {my_hosts_idx}
            else:
                # Not controlled and Host does not have any service yet
                unknown_hosts.add(host)
                # The unknown do not change concept so they are all together
                unknown_hosts_idx = IP(unknown_hosts_concept)

                concept_mapping['known_hosts'][unknown_hosts_idx] = unknown_hosts
                try:
                    ip_to_concept[host].add(unknown_hosts_idx)
                except KeyError:
                    ip_to_concept[host] = {unknown_hosts_idx}

    # Convert Services
    # state.known_services: dict. {'ip': Service object}
    #   Service - Object
    #       service.name 'openssh' (what in the configuration is 'type'). 
    #       service.type 'passive' | 'active' (This is added by our env when finding the )
    #       service.version : '1.1.1.'
    #       service.is_local: Bool

    # The problem here is that the ip gets changed to some concept, but then the concept aggregates many ips, and when you want to exploit them, you don't know which one was anymore.
                # One solution is to change the concept to something like 'unknown1' and add a small changer. 
                # This would assign a unique concept to each IP, which breaks the 'getting things together' part, but maybe is fine.
                # What are the implicances of changing '192.168.2.4' to 'remote3'? I think the idea is that any ip in the future can get assigned to this concept
                # which means that it can generalize.
    for ip, service in state.known_services.items():
        concepts_host_idx = ip_to_concept[ip]
        for concept_host_idx in concepts_host_idx:
            concept_mapping['known_services'][concept_host_idx] = service

    # Convert Data
    # state.known_data: dict. {'ip': Data object}
    #   Data - Object
    #       data.ownwer
    #       data.id
    for ip, data in state.known_data.items():
        concepts_host_idx = ip_to_concept[ip]
        for concept_host_idx in concepts_host_idx:
            concept_mapping['known_data'][concept_host_idx] = data

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

    logger.info(f"\tI2C: New concept known_hosts: {concept_mapping['known_hosts']}")
    logger.info(f"\tI2C: New concept controlled_hosts: {concept_mapping['controlled_hosts']}")
    logger.info(f"\tI2C: New concept known_nets: {concept_mapping['known_networks']}")
    logger.info(f"\tI2C: New concept known_services: {concept_mapping['known_services']}")
    logger.info(f"\tI2C: New concept known_data: {concept_mapping['known_data']}")

    # Prepare to return concepts
    state_controlled_hosts = {host for host in concept_mapping['controlled_hosts']}
    state_known_hosts = {host for host in concept_mapping['known_hosts']}
    state_networks = {net for net in concept_mapping['known_networks']}
    state_known_services = concept_mapping['known_services']
    state_known_data = concept_mapping['known_data']

    # TODO: Check what happens when the concepts get empty. If there are no more unknown hosts, we should delete that 

    new_state = GameState(state_controlled_hosts, state_known_hosts, state_known_services, state_known_data, state_networks)
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
        # parameters = {
        # "target_host": IP(parameters_dict["target_host"]["ip"]), 
        # "target_service": Service(
        #   parameters_dict["target_service"]["name"], 
        #   parameters_dict["target_service"]["type"], 
        #   parameters_dict["target_service"]["version"], 
        #   parameters_dict["target_service"]["is_local"]), 
        # "source_host": IP(parameters_dict["source_host"]["ip"])}

        # Change the target host from concept to ip
        target_host_concept = action.parameters['target_host']
        for host_concept in concept_mapping['known_hosts']:
            if target_host_concept == host_concept and 'unknown' in host_concept.ip:
             new_target_host = random.choice(list(concept_mapping['known_hosts'][host_concept]))
            elif target_host_concept == host_concept:
                new_target_host = concept_mapping['known_hosts'][host_concept]

        # Service is not changed for now
        target_service_concept = action.parameters['target_service']
        new_target_service = target_service_concept

        # Change the src host from concept to ip
        src_host_concept = action.parameters['source_host']
        for host_concept in concept_mapping['controlled_hosts']:
            if src_host_concept == host_concept:
                new_src_host = concept_mapping['controlled_hosts'][host_concept]

        action = Action(ActionType.ExploitService, parameters={"target_host": new_target_host, "target_service": new_target_service, "source_host": new_src_host})

    elif action._type == ActionType.ExfiltrateData:
        # parameters = {
        # "target_host": IP(parameters_dict["target_host"]["ip"]), 
        # "source_host": IP(parameters_dict["source_host"]["ip"]), 
        # "data": Data(parameters_dict["data"]["owner"],parameters_dict["data"]["id"])}

        # Change the target host from concept to ip
        target_host_concept = action.parameters['target_host']
        for host_concept in concept_mapping['controlled_hosts']:
            #if target_host_concept == host_concept and 'unkown' in host_concept.ip:
                #new_target_host = random.choice(list(concept_mapping['known_hosts'][host_concept]))
                #break
            if target_host_concept == host_concept:
                new_target_host = concept_mapping['controlled_hosts'][host_concept]
                break

        # Change the src host from concept to ip
        src_host_concept = action.parameters['source_host']
        for host_concept in concept_mapping['controlled_hosts']:
            if src_host_concept == host_concept:
                new_src_host = concept_mapping['controlled_hosts'][host_concept]

        # TODO: Change dat
        data_concept = action.parameters['data']
        new_data = data_concept

        action = Action(ActionType.ExfiltrateData, parameters={"target_host": new_target_host, "source_host": new_src_host, "data": new_data})

    elif action._type == ActionType.FindData:
        # parameters = {
        # "source_host": IP(parameters_dict["source_host"]["ip"]), 
        # "target_host": IP(parameters_dict["target_host"]["ip"])}

        # Change the target host from concept to ip
        target_host_concept = action.parameters['target_host']
        for host_concept in concept_mapping['controlled_hosts']:
            #if target_host_concept == host_concept and 'unknown' in host_concept.ip:
                #new_target_host = random.choice(list(concept_mapping['controlled_hosts'][host_concept]))
            if target_host_concept == host_concept:
                new_target_host = concept_mapping['controlled_hosts'][host_concept]

        # Change the src host from concept to ip
        src_host_concept = action.parameters['source_host']
        for host_concept in concept_mapping['controlled_hosts']:
            if src_host_concept == host_concept:
                new_src_host = concept_mapping['controlled_hosts'][host_concept]
        
        action = Action(ActionType.FindData, parameters={"target_host": new_target_host, "source_host": new_src_host})

    elif action._type == ActionType.ScanNetwork:
        target_net_concept = action.parameters['target_network']
        for net_concept in concept_mapping['known_networks']:
            if target_net_concept == net_concept and 'unknown' in net_concept.ip:
                new_target_network = random.choice(list(concept_mapping['known_networks'][net_concept]))
                break
            elif target_net_concept == net_concept:
                new_target_network = concept_mapping['known_networks'][net_concept]
                break
        # Change the src host from concept to ip
        src_host_concept = action.parameters['source_host']
        for host_concept in concept_mapping['controlled_hosts']:
            if src_host_concept == host_concept:
                new_src_host = concept_mapping['controlled_hosts'][host_concept]
        action = Action(ActionType.ScanNetwork, parameters={"source_host": new_src_host, "target_network": new_target_network} )

    elif action._type == ActionType.FindServices:
        # parameters = {
        # "source_host": IP(parameters_dict["source_host"]["ip"]), 
        # "target_host": IP(parameters_dict["target_host"]["ip"])}

        # Change the target host from concept to ip
        target_host_concept = action.parameters['target_host']
        for host_concept in concept_mapping['known_hosts']:
            if target_host_concept == host_concept and 'unknown' in host_concept.ip:
                new_target_host = random.choice(list(concept_mapping['known_hosts'][host_concept]))
            elif target_host_concept == host_concept:
                new_target_host = concept_mapping['known_hosts'][host_concept]

        # Change the src host from concept to ip
        src_host_concept = action.parameters['source_host']
        for host_concept in concept_mapping['controlled_hosts']:
            if src_host_concept == host_concept:
                new_src_host = concept_mapping['controlled_hosts'][host_concept]
        action = Action(ActionType.FindServices, parameters={"source_host": new_src_host, "target_host": new_target_host} )

    return action