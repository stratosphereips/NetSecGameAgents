"""
Collection of functions which are intended for agents to share.
Functionality which is specific for a certain type of agent should be implementented
directly in the agent class. 

author: Ondrej Lukas - ondrej.lukas@aic.fel.cvut.cz
"""
from collections import namedtuple
import random
import ipaddress
from AIDojoCoordinator.game_components import Action, ActionType, GameState, Observation, IP, Network

def generate_valid_actions_concepts(state: GameState, include_blocks=False)->list:
    """
    Function that generates a list of all valid actions in a given state for the conceptual agents.
    """

    def is_fw_blocked(state, source_host, target_host)->bool:
        blocked = False
        try:
            blocked = target_host in state.known_blocks[source_host]
        except KeyError:
            pass #this src ip has no known blocks
        return blocked 

    valid_actions = set()

    for source_host in state.controlled_hosts:
        # Network Scans
        for network in state.known_networks:
            # TODO ADD neighbouring networks
            # Only scan local from local hosts
            # If the network or source_host are str and not 'external', they are concepts and also private.
            if ( 
                (
                    type(network.ip) is str  
                    and 'external' not in network.ip
                    and type(source_host) is str 
                    and 'external' not in source_host
                ) or (
                    type(network.ip) is Network
                    network.is_private() 
                    and source_host.is_private()
                    )
            ): 
                valid_actions.add(Action(ActionType.ScanNetwork, parameters={"target_network": network, "source_host": source_host,}))

        # Service Scans
        for target_host in state.known_hosts:
            # Do not scan a service from an external host
            # Do not scan a service in an external host
            # If the network or source_host are str, they are concepts and also private.
            # Do not scan the target host if it is blocked in that source host
            # Do not scan the source host if it is blocked in that target host
            # Do not service scan the same host you are in
            if (
                (
                    (
                        type(target_host) is str 
                        and 'external' not in target_host 
                        and type(source_host) is str
                        and 'external' not in source_host
                    ) or (
                        network.is_private() 
                        and source_host.is_private()
                    )
                ) and (
                    not is_fw_blocked(state, source_host, target_host)
                    and not is_fw_blocked(state, target_host, source_host)
                    and target_host != source_host
                )
            ): 
                valid_actions.add(Action(ActionType.FindServices, parameters={"target_host": target_host, "source_host": source_host,}))

        # Service Exploits
        for target_host, service_list in state.known_services.items():
            # Do not exploit a service from an external host
            # Do not exploit a service in an external host
            # If the network or source_host are str, they are concepts and also private.
            # Only exploits target_hosts we do not control
            # Do not exploit the target host if it is blocked in that source host
            # Do not exploit the source host if it is blocked in that target host
            # Do not exploit the same host you are in
            if (
                (
                    (
                        type(target_host) is str 
                        and 'external' not in target_host 
                        and type(source_host) is str 
                        and 'external' not in source_host
                    ) or (
                        network.is_private() 
                        and source_host.is_private()
                    )
                ) and (
                    target_host not in state.controlled_hosts
                    and not is_fw_blocked(state, source_host, target_host)
                    and not is_fw_blocked(state, target_host, source_host)
                    and target_host != source_host
                )
            ):
                for service in service_list:
                    # Do not consider local services, which are internal to the target_host
                    if not service.is_local:
                        valid_actions.add(Action(ActionType.ExploitService, parameters={"target_host": target_host,"target_service": service,"source_host": source_host,}))

        # Find Data Scans
        for target_host in state.controlled_hosts:
            # Do not find data from external hosts
            # Do not find data in external hosts
            # Only find data in hosts we control (implicit from the source of data)
            # Do not find data in the target host if it is blocked in that source host
            # Do not find data in the source host if it is blocked in that target host
            # Do not find data in the same host you are in
            if (
                (
                        (
                        type(target_host) is str 
                        and 'external' not in target_host 
                        and type(source_host) is str 
                        and 'external' not in source_host
                    ) or (
                        network.is_private() 
                        and source_host.is_private()
                    )
                ) and (
                    target_host in state.controlled_hosts
                    and not is_fw_blocked(state, source_host, target_host)
                    and not is_fw_blocked(state, target_host, source_host)
                    and target_host != source_host
                )
            ): 
                valid_actions.add(Action(ActionType.FindData, parameters={"target_host": target_host, "source_host": source_host}))

        # Data Exfiltration
        for source_host, data_list in state.known_data.items():
            # Do not exfiltrate data from external hosts
            # Do not exfiltrate data to internal hosts
            # Only exfiltrate data from hosts we control (implicit from the source of data)
            # Only exfiltrate data to hosts we control
            # Only exfiltrate data from hosts with data
            # Only exfiltrate data to hosts we control
            # Do not exfiltrate to and from the same host
            # Do not exfiltarte to the target host if it is blocked in that source host
            # Do not exfiltarte to the source host if it is blocked in that target host
            # Do not exfiltrate to the same host you are in
            if (
                (
                        (
                        type(target_host) is str 
                        and 'external' in target_host # Controversial rule since some agents may choose to temporarily exfiltrate to an internal host to avoid detection
                        and type(source_host) is str 
                        and 'external' not in source_host
                    ) or (
                        network.is_private() 
                        and source_host.is_private()
                    )
                ) and (
                    target_host in state.controlled_hosts
                    and not is_fw_blocked(state, source_host, target_host)
                    and not is_fw_blocked(state, target_host, source_host)
                    and target_host != source_host
                ) and (
                    data_list is not None
                )
            ): 
                for data in data_list:
                    for target_host in state.controlled_hosts:
                        if target_host != source_host:
                            valid_actions.add(Action(ActionType.ExfiltrateData, parameters={"target_host": target_host, "source_host": source_host, "data": data}))

        # BlockIP
        if include_blocks:
            # Explanation of action
            # The target host is the host where the blocking will be applied (the FW)
            # The source host is the host that the agent uses to connect to the target host. A host that must be controlled by the agent
            # The blocked host is the host that will be included in the FW list to be blocked.

            # Do not block external hosts
            # Do not block internal hosts from external hosts
            # Do not block if the source host is not controlled
            # Do not block if the target host is not controlled
            # Do not block if the combination of source, and target host, and blocked host is already blocked
            # Do not block the same host you are in
            if (
                (
                        (
                        type(target_host) is str 
                        and 'external' not in target_host 
                        and type(source_host) is str 
                        and 'external' not in source_host
                    ) or (
                        network.is_private() 
                        and source_host.is_private()
                    )
                ) and (
                    (
                    target_host in state.controlled_hosts
                    and source_host in state.controlled_hosts # these are verified also below in the for
                    and blocked_host != source_host
                    )
                )
            ): 
                for source_host in state.controlled_hosts:
                    for target_host in state.controlled_hosts:
                        if not is_fw_blocked(state, source_host, target_host):
                            for blocked_host in state.known_hosts:
                                valid_actions.add(Action(ActionType.BlockIP, {"target_host":target_host, "source_host":source_host, "blocked_host":blocked_host}))
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

    for source_host in state.controlled_hosts:
        #Network Scans
        for network in state.known_networks:
            # TODO ADD neighbouring networks
            valid_actions.add(Action(ActionType.ScanNetwork, parameters={"target_network": network, "source_host": source_host,}))

        # Service Scans
        for blocked_host in state.known_hosts:
            if not is_fw_blocked(state, source_host, blocked_host):
                valid_actions.add(Action(ActionType.FindServices, parameters={"target_host": blocked_host, "source_host": source_host,}))

        # Service Exploits
        for blocked_host, service_list in state.known_services.items():
            if not is_fw_blocked(state, source_host,blocked_host):
                for service in service_list:
                    valid_actions.add(Action(ActionType.ExploitService, parameters={"target_host": blocked_host,"target_service": service,"source_host": source_host,}))
        # Data Scans
        for blocked_host in state.controlled_hosts:
            if not is_fw_blocked(state, source_host,blocked_host):
                valid_actions.add(Action(ActionType.FindData, parameters={"target_host": blocked_host, "source_host": blocked_host}))

        # Data Exfiltration
        for source_host, data_list in state.known_data.items():
            for data in data_list:
                for trg_host in state.controlled_hosts:
                    if trg_host != source_host:
                        if not is_fw_blocked(state, source_host,trg_host):
                            valid_actions.add(Action(ActionType.ExfiltrateData, parameters={"target_host": trg_host, "source_host": source_host, "data": data}))
        
        # BlockIP
        if include_blocks:
            for source_host in state.controlled_hosts:
                for target_host in state.controlled_hosts:
                    if not is_fw_blocked(state, source_host,target_host):
                        for blocked_ip in state.known_hosts:
                            valid_actions.add(Action(ActionType.BlockIP, {"target_host":target_host, "source_host":source_host, "blocked_host":blocked_ip}))
    return list(valid_actions)    

def _format_dict_section(section_dict, section_name):
    """
    Helper function to format a dictionary section for state string representation.

    Args:
        section_dict: Dictionary to format
        section_name: Name of the section

    Returns:
        Formatted string for the section
    """
    result = f"{section_name}:{{"
    for host in sorted(section_dict.keys()):
        result += f"{host}:[{','.join([str(x) for x in sorted(section_dict[host])])}]"
    result += "}"
    return result


def state_as_ordered_string(state:GameState)->str:
    """Function for generating string representation of a SORTED gamestate components. Can be used as key for dictionaries."""
    ret = ""
    ret += f"nets:[{','.join([str(x) for x in sorted(state.known_networks)])}],"
    ret += f"hosts:[{','.join([str(x) for x in sorted(state.known_hosts)])}],"
    ret += f"controlled:[{','.join([str(x) for x in sorted(state.controlled_hosts)])}],"
    ret += _format_dict_section(state.known_services, "services") + ","
    ret += _format_dict_section(state.known_data, "data") + ","
    ret += _format_dict_section(state.known_blocks, "blocks")
    return ret

def recompute_reward(observation: Observation) -> Observation:
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

def _categorize_host_by_service(host, service, counter, host_words, host_concept, concept_mapping, ip_to_concept, unknown_hosts, state):
    """
    Helper function to categorize a host based on service keywords.

    Args:
        host: The host to categorize
        service: The service object
        counter: Counter for unique concept naming
        host_words: List of keywords to match against service name
        host_concept: The concept name (e.g., 'db', 'web', 'remote', 'files')
        concept_mapping: The mapping dictionary
        ip_to_concept: Reverse mapping from IP to concepts
        unknown_hosts: Set of unknown hosts
        state: The game state

    Returns:
        True if host was categorized, False otherwise
    """
    for word in host_words:
        if word in service.name:
            host_idx = IP(host_concept + str(counter))
            concept_mapping['known_hosts'][host_idx] = host
            try:
                ip_to_concept[host].add(host_idx)
            except KeyError:
                ip_to_concept[host] = {host_idx}
            unknown_hosts.discard(host)
            # Is it also controlled?
            if host in state.controlled_hosts:
                concept_mapping['controlled_hosts'][host_idx] = host
            return True  # Found a match, break out of word loop
    return False  # No match found


def convert_ips_to_concepts(observation, logger):
    """
    New ideas
    - All the hosts that can not be distinguished should be together in one concept
    - As soon as something distinguishes them, change the name to that, like Host22 for a host with port 22
    - For known ports, does it make sense to say HostDB? The problem is that all db ports will be confused. Is it better Host3306?
    - Also mark hosts when they were already scanned and no port was there, like HostClosed
    - The internet hosts should be HostInternet (so that the agent can distinguish them)
    - The host where you start should be HostStarting (so that the agent can distinguish them)
    - The main problem is that even for a human it can be impossible to transfer knowledge from one net to the other. So for sure you need to explore in a new network. However, humans do not explore like idiots agents because they have more information about the network without needing to scan hosts one by one. They can check the configurations, processes, mem, and network traffic to help them selves.

    Function to convert the IPs and networks in the observation into a concept 
    so the agent is not dependent on IPs and specific values
    in: observation with IPs
    out: observation with concepts, dict with concept_mapping
    """
    # observation.controlled_hosts: set
    # observation.known_hosts: set
    # observation.known_networks: set
    # observation.known_data: dict. {'ip': Data object}
    #   Data - Object
    #       data.ownwer
    #       data.id
    # observation.known_services: dict. {'ip': Service object}
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

    # Here we keep the mapping of concepts to values such as IPs and Networks
    concept_mapping = {'controlled_hosts': {}, 'known_hosts': {}, 'known_services': {}, 'known_data': {}, 'known_networks': {}}

    #######
    # Hosts Concepts to Use

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


    ##########################
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

    ##########################
    # Convert the known hosts
    logger.info(f'\tI2C: Converting known hosts')
    logger.info(f'\tI2C: Real state known hosts: {state.known_hosts}')
    logger.info(f'\tI2C: Real state controlled hosts: {state.controlled_hosts}')
    # Counter to separate concepts in same category
    counter = 0
    for host in state.known_hosts:
        counter += 1
        # Is it external and it is not in ip_to_concept, so it is not controlled
        if not host.is_private():
            external_hosts_idx = external_hosts_concept+str(counter)
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
                # Try to categorize the host based on its services
                _categorize_host_by_service(host, service, counter, db_hosts_words, db_hosts_concept, concept_mapping, ip_to_concept, unknown_hosts, state)
                _categorize_host_by_service(host, service, counter, web_hosts_words, web_hosts_concept, concept_mapping, ip_to_concept, unknown_hosts, state)
                _categorize_host_by_service(host, service, counter, remote_hosts_words, remote_hosts_concept, concept_mapping, ip_to_concept, unknown_hosts, state)
                _categorize_host_by_service(host, service, counter, files_hosts_words, files_hosts_concept, concept_mapping, ip_to_concept, unknown_hosts, state)
        else:
            # These are all the devices without services
            # A device can be controlled (specially in the first state from the env)
            if host in state.controlled_hosts:
                # Yes it is controlled
                my_hosts_idx = my_hosts_concept+str(counter)
                concept_mapping['controlled_hosts'][my_hosts_idx] = host
                concept_mapping['known_hosts'][my_hosts_idx] = host
                ip_to_concept[host] = {my_hosts_idx}
            else:
                # Not controlled and Host does not have any service yet
                unknown_hosts.add(host)
                # The unknown do not change concept so they are all together
                unknown_hosts_idx = unknown_hosts_concept

                concept_mapping['known_hosts'][unknown_hosts_idx] = unknown_hosts
                try:
                    ip_to_concept[host].add(unknown_hosts_idx)
                except KeyError:
                    ip_to_concept[host] = {unknown_hosts_idx}

    ##########################
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

    ##########################
    # Convert Data
    # state.known_data: dict. {'ip': Data object}
    #   Data - Object
    #       data.ownwer
    #       data.id
    for ip, data in state.known_data.items():
        concepts_host_idx = ip_to_concept[ip]
        for concept_host_idx in concepts_host_idx:
            concept_mapping['known_data'][concept_host_idx] = data

    ##########################
    # Convert Networks
    # The set for my networks. Networks here you control a host
    my_networks = set()
    # The index object
    my_nets = Network('mynet', 24)

    # The set
    unknown_networks = set()
    # The index object
    unknown_nets = Network('unknown', 24)

    for network in state.known_networks:
        # net_assigned is only to speed up the process when a network has been added because the agent
        # controlls a host there, or two.
        net_assigned = False

        # Find the mynet networks
        for controlled_ip in state.controlled_hosts:
            if ipaddress.IPv4Address(controlled_ip.ip) in ipaddress.IPv4Network(f'{network.ip}/{network.mask}'):
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
    # Create a new namedtuple with the common observation and the new concept mapping, so we can pass it around.
    new_observation = namedtuple('ConceptObservation', ['observation', 'concept_mapping'])
    new_observation = new_observation(Observation(new_state, reward, end, info), concept_mapping)
    return new_observation

def _convert_target_host_concept_to_ip(target_host_concept, concept_observation, use_controlled_hosts=False):
    """
    Helper function to convert target host concept to IP.

    Args:
        target_host_concept: The concept host to convert
        concept_observation: The observation that now has concepts
        use_controlled_hosts: If True, search in controlled_hosts, otherwise in known_hosts

    Returns:
        The corresponding IP address
    """
    host_mapping_set = concept_observation.state.controlled_hosts if use_controlled_hosts else concept_observation.state.known_hosts

    # Check if the target host concept exists in the mapping
    if target_host_concept in host_mapping_set:
        mapped_value = host_mapping_set[target_host_concept]
        if 'unknown' in str(target_host_concept):
            # For unknown hosts, choose randomly from the mapped set/value
            if isinstance(mapped_value, set):
                return random.choice(list(mapped_value))
            else:
                return mapped_value
        else:
            return mapped_value

    # Fallback: return a random choice from available hosts
    if host_mapping_set:
        return random.choice(list(host_mapping_set.values()))
    return target_host_concept


def _convert_source_host_concept_to_ip(src_host_concept, concept_observation):
    """
    Helper function to convert source host concept to IP.
    Source hosts are always from controlled_hosts.

    Args:
        src_host_concept: The concept host to convert
        concept_observation: The concept observstion mapping containing concept to IP mappings

    Returns:
        The corresponding IP address
    """
    controlled_hosts_dict = concept_observation.state.controlled_hosts

    # Check if the source host concept exists in the mapping
    if src_host_concept in controlled_hosts_dict:
        return controlled_hosts_dict[src_host_concept]

    # Fallback: return a random controlled host if no exact match found
    if controlled_hosts_dict:
        return random.choice(list(controlled_hosts_dict.values()))
    return src_host_concept


def _convert_network_concept_to_ip(target_net_concept, concept_observation):
    """
    Helper function to convert network concept to actual network.

    Args:
        target_net_concept: The concept network to convert
        concept_observation: The concept observstion mapping containing concept to Network mappings

    Returns:
        The corresponding network object
    """
    networks_dict = concept_observation.state.known_networks

    # Check if the target network concept exists in the mapping
    if target_net_concept in networks_dict:
        mapped_value = networks_dict[target_net_concept]
        if 'unknown' in str(target_net_concept):
            # For unknown networks, choose randomly from the mapped set/value
            if isinstance(mapped_value, set):
                return random.choice(list(mapped_value))
            else:
                return mapped_value
        else:
            # For known networks, the mapped value might be a set of networks
            if isinstance(mapped_value, set):
                return random.choice(list(mapped_value))
            else:
                return mapped_value

    # Fallback: return a random network if no exact match found
    if networks_dict:
        all_networks = []
        for net_set in networks_dict.values():
            if isinstance(net_set, set):
                all_networks.extend(list(net_set))
            else:
                all_networks.append(net_set)
        return random.choice(all_networks) if all_networks else target_net_concept
    return target_net_concept


def convert_concepts_to_actions(action, observation):
    """
    Function to convert the concepts learned before into IPs and networks
    so the env knows where to really act
    in: action for concepts
    in: state with concepts
    out: action for IPs
    """
    if action.type == ActionType.ExploitService:
        # Convert target and source hosts using helper functions
        new_target_host = _convert_target_host_concept_to_ip(action.parameters['target_host'], observation)
        new_src_host = _convert_source_host_concept_to_ip(action.parameters['source_host'], observation)

        # Service is not changed for now
        new_target_service = action.parameters['target_service']

        action = Action(ActionType.ExploitService, parameters={
            "target_host": new_target_host,
            "target_service": new_target_service,
            "source_host": new_src_host
        })

    elif action.type == ActionType.ExfiltrateData:
        # Convert target and source hosts using helper functions (both from controlled hosts for exfiltration)
        new_target_host = _convert_target_host_concept_to_ip(action.parameters['target_host'], observation, use_controlled_hosts=True)
        new_src_host = _convert_source_host_concept_to_ip(action.parameters['source_host'], observation)

        # Data is not changed for now
        new_data = action.parameters['data']

        action = Action(ActionType.ExfiltrateData, parameters={
            "target_host": new_target_host,
            "source_host": new_src_host,
            "data": new_data
        })

    elif action.type == ActionType.FindData:
        # Convert target and source hosts using helper functions (both from controlled hosts for FindData)
        new_target_host = _convert_target_host_concept_to_ip(action.parameters['target_host'], observation, use_controlled_hosts=True)
        new_src_host = _convert_source_host_concept_to_ip(action.parameters['source_host'], observation)

        action = Action(ActionType.FindData, parameters={
            "target_host": new_target_host,
            "source_host": new_src_host
        })

    elif action.type == ActionType.ScanNetwork:
        # Convert network and source host using helper functions
        new_target_network = _convert_network_concept_to_ip(action.parameters['target_network'], observation)
        new_src_host = _convert_source_host_concept_to_ip(action.parameters['source_host'], observation)

        action = Action(ActionType.ScanNetwork, parameters={
            "source_host": new_src_host,
            "target_network": new_target_network
        })

    elif action.type == ActionType.FindServices:
        # Convert target and source hosts using helper functions
        new_target_host = _convert_target_host_concept_to_ip(action.parameters['target_host'], observation)
        new_src_host = _convert_source_host_concept_to_ip(action.parameters['source_host'], observation)

        action = Action(ActionType.FindServices, parameters={
            "source_host": new_src_host,
            "target_host": new_target_host
        })

    return action