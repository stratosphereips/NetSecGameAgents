"""
Collection of functions which are intended for agents to share.
Functionality which is specific for a certain type of agent should be implementented
directly in the agent class. 

author: Ondrej Lukas - ondrej.lukas@aic.fel.cvut.cz
"""
from collections import namedtuple
import random
import ipaddress
import sys
from AIDojoCoordinator.game_components import Action, ActionType, GameState, Observation, Network, AgentStatus

def generate_valid_actions_concepts(state: GameState, action_history: set, include_blocks=False)->list:
    """
    Function that generates a list of all valid actions in a given state for the conceptual agents.
    It receives the concepts_acted_on set that contains the concepts that have been already acted on. To avoid 
    acting on the same concept twice.
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
                type(network.ip) is str  
                and 'external' not in network.ip
                and type(source_host) is str 
                and 'external' not in source_host
            ): 
                action = Action(ActionType.ScanNetwork, parameters={"target_network": network, "source_host": source_host,})
                # Check if the action is in the history of actions
                if action not in action_history:
                    valid_actions.add(action)

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
                    type(target_host) is str 
                    and 'external' not in target_host 
                    and type(source_host) is str
                    and 'external' not in source_host
                ) and (
                    not is_fw_blocked(state, source_host, target_host)
                    and not is_fw_blocked(state, target_host, source_host)
                    # And target_host has no services, so we dont search for services if we have them
                    and target_host not in state.known_services
                )
            ): 
                action = Action(ActionType.FindServices, parameters={"target_host": target_host, "source_host": source_host,})
                # Check if the action is in the history of actions
                if action not in action_history:
                    valid_actions.add(action)

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
                    type(target_host) is str 
                    and 'external' not in target_host 
                    and type(source_host) is str 
                    and 'external' not in source_host
                ) and (
                    target_host not in state.controlled_hosts
                    and not is_fw_blocked(state, source_host, target_host)
                    and not is_fw_blocked(state, target_host, source_host)
                    and target_host != source_host
                    # and target_host is not controlled host so we dont re exploit an exploited host from any source
                    and target_host not in state.controlled_hosts
                )
            ):
                for service in service_list:
                    # Do not consider local services, which are internal to the target_host
                    if not service.is_local:
                        action = Action(ActionType.ExploitService, parameters={"target_host": target_host,"target_service": service,"source_host": source_host,})
                        # Check if the action is in the history of actions
                        if action not in action_history:
                            valid_actions.add(action)

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
                    type(target_host) is str 
                    and 'external' not in target_host 
                    and type(source_host) is str 
                    and 'external' not in source_host
                ) and (
                    target_host in state.controlled_hosts
                    and not is_fw_blocked(state, source_host, target_host)
                    and not is_fw_blocked(state, target_host, source_host)
                    # And target_host does not has currently data
                    and target_host not in state.known_data
                )
            ): 
                action = Action(ActionType.FindData, parameters={"target_host": target_host, "source_host": source_host})
                # Check if the action is in the history of actions
                if action not in action_history:
                    valid_actions.add(action)

        # Data Exfiltration
        for source_host, data_list in state.known_data.items():
            for target_host in state.controlled_hosts:
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
                # Do not exfiltrate if the target_host already has the data
                if (
                    (
                        type(source_host) is str 
                        and 'external' not in source_host
                    ) and (
                        target_host in state.controlled_hosts
                        and not is_fw_blocked(state, source_host, target_host)
                        and target_host != source_host
                    ) and (
                        data_list is not None
                        and target_host != source_host
                    )
                ): 
                    for data in data_list:
                        # This check not to exfiltrate the data several times is more organic here
                        # checking if the data is already in the target controlled host
                        data_was_not_exfiltrated_before = True
                        ignore_this_data = False

                        # Should some type of data be ignored?
                        # Ignore logfiles
                        if data.id == 'logfile':
                            ignore_this_data = True
                            continue

                        try:
                            # Is the target_host in the list of known_hosts?
                            _ = state.known_data[target_host]
                            for exfiltrated_data in state.known_data[target_host]: 
                                # Does the target_host already have the data?
                                if exfiltrated_data.id == data.id:
                                    data_was_not_exfiltrated_before = False
                        except KeyError:
                            # We dont have data in this target host so is ok
                            pass

                        action = Action(ActionType.ExfiltrateData, parameters={"target_host": target_host, "source_host": source_host, "data": data})
                        # Check if the action is in the history of actions
                        if not ignore_this_data and data_was_not_exfiltrated_before and action not in action_history:
                            valid_actions.add(action)


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
                    type(target_host) is str 
                    and 'external' not in target_host 
                    and type(source_host) is str 
                    and 'external' not in source_host
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
                                # Check if the action is in the history of actions
                                action = Action(ActionType.BlockIP, {"target_host":target_host, "source_host":source_host, "blocked_host":blocked_host})
                                if action not in action_history:
                                    valid_actions.add(action)
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
    state = observation.state
    reward = observation.reward
    end = observation.end
    info = observation.info

    # The rewards hare are the originals from the env. 
    # Each agent can do this differently
    if info and info['end_reason'] == AgentStatus.Fail:
        reward = -100
    elif info and info['end_reason'] == AgentStatus.Success:
        reward = 100
    elif info and info['end_reason'] == AgentStatus.TimeoutReached:
        reward = -1
    else:
        reward = -1
    
    new_observation = Observation(GameState.from_dict(state), reward, end, info)
    return new_observation

def convert_ips_to_concepts(observation, logger, concept_logger=None):
    """
    Function to convert the IPs and networks in the observation into a concept 
    so the agent is not dependent on IPs and specific values

    in: observation with IPs
    out: observation with concepts, dict with concept_mapping

    observation.controlled_hosts: set
    observation.known_hosts: set
    observation.known_networks: set
    observation.known_data: dict. {'ip': Data object}
      Data - Object
          data.ownwer
          data.id
    observation.known_services: dict. {'ip': Service object}
      Service - Object
          service.name 'openssh' (what in the configuration is 'type'). The service name is derived from the nmap services https://svn.nmap.org/nmap/nmap-services
          service.type 'passive' | 'active' (This is added by our env when finding the )
          service.version : '1.1.1.'
          service.is_local: Bool
    """

    new_observation = None
    state = observation.state
    reward = observation.reward
    end = observation.end
    info = observation.info

    # Here we keep the mapping of concepts to values such as IPs and Networks
    concept_mapping = {'controlled_hosts': {}, 'known_hosts': {}, 'known_services': {}, 'known_data': {}, 'known_networks': {}}

    #######
    # Hosts Concepts to Use

    # Host are separated according to their location (external or internal) and their ports (services)
    external_hosts_concept = 'external'
    external_hosts_concept_counter = 0
    unknown_hosts_concept = 'unknown'
    priv_hosts_concept = 'host'
    priv_hosts_concept_counter = 0
    unknown_hosts_concept_counter = 0
    network_concept_counter = 0
   

    # To have a reverse dict of ips to concepts so we can assign the controlled hosts fast from the known hosts
    # This is a performance trick
    # The ip can have one concept ip_to_concept['1.1.1.1'] = 'web'
    ip_to_concept = {}

    # Log the real hosts before the modification
    logger.info(f'\tI2C: Real state known nets: {state.known_networks}')
    logger.info(f'\tI2C: Real state known hosts: {state.known_hosts}')
    logger.info(f'\tI2C: Real state controlled hosts: {state.controlled_hosts}')
    logger.info(f'\tI2C: Real state known services: {state.known_services}')
    logger.info(f'\tI2C: Real state known data: {state.known_data}')
    logger.info(f'\tI2C: Real state known blocks: {state.known_blocks}')

    ##########################
    # Convert controlled hosts    
    # We do not convert the controlled hosts directly because they are converted when we do
    # known_hosts and services and data. In that way we can keep track of the correct counters. 
    # So no need to do it separated here.

     ##########################
    # Convert the known hosts
    # Counter to separate concepts in same category
    for host in state.known_hosts:
        # If the host is external
        if not host.is_private():
            external_hosts_idx = external_hosts_concept + str(external_hosts_concept_counter)
            external_hosts_concept_counter += 1
            concept_mapping['known_hosts'][external_hosts_idx] = host
            ip_to_concept[host] = external_hosts_idx
            # Is it also controlled?
            if host in state.controlled_hosts:
                concept_mapping['controlled_hosts'][external_hosts_idx] = host
            continue
        elif host in state.controlled_hosts:
            # Yes it is controlled. So it is not unknown
            # Host is internal 
            privnet_hosts_idx = priv_hosts_concept + str(priv_hosts_concept_counter)
            priv_hosts_concept_counter += 1
            concept_mapping['controlled_hosts'][privnet_hosts_idx] = host
            concept_mapping['known_hosts'][privnet_hosts_idx] = host
            ip_to_concept[host] = privnet_hosts_idx
        else:
            # The host is not controlled, so it is unknown
            unknown_hosts_idx = unknown_hosts_concept + str(unknown_hosts_concept_counter)
            concept_mapping['known_hosts'][unknown_hosts_idx] = host
            ip_to_concept[host] = unknown_hosts_idx
            unknown_hosts_concept_counter += 1

    ##########################
    # Convert Services
    # state.known_services: dict. {'ip': Service object}
    #   Service - Object
    #       service.name '22/tcp, openssh' (what in the configuration is 'type'). 
    #       service.type 'passive' | 'active' (This is added by our env when finding the )
    #       service.version : '1.1.1.'
    #       service.is_local: Bool (if the service is only for localhost (True) or external (False))
    # It is ok to add the services one by one in the concept even if the were before all together in a set() in the real state
    # because later when the actions are created each service is used for a new action.

    for ip, services in state.known_services.items():
        # Get port numbers
        port_numbers = ''
        for service in services:
            # Ignore local services since can not be attacked from the outside
            if not service.is_local:
                try:
                    port_numbers += '_' + service.name.split(", ")[0]
                except IndexError:
                    port_numbers = 'NoPort'

        # Now deal with all the services together

        # If it was before in ip to concepts, delete for now  so we can change the name
        try:
            prev_concepts_host_idx = ip_to_concept[ip]
            # Now remove the past name
            ip_to_concept.pop(ip)
        except KeyError:
            # We dont have it. ok.
            pass

        # All hosts that have some ports are called 'host' + something.
        concepts_host_idx = 'host' + str(priv_hosts_concept_counter)
        priv_hosts_concept_counter += 1

        # Add the ports to the host concept 
        new_concepts_host_idx = f'{concepts_host_idx}{port_numbers}'
        # Change the old concept name to the new one
        concept_mapping['known_services'][new_concepts_host_idx] = services

        # Add to ip_to_concept
        ip_to_concept[ip] = new_concepts_host_idx

        # Check if that old concept was in known_hosts and delete it
        try:
            _ = concept_mapping['known_hosts'][prev_concepts_host_idx]
            # Rename it from known hosts
            del concept_mapping['known_hosts'][prev_concepts_host_idx]
            concept_mapping['known_hosts'][new_concepts_host_idx] = ip
        except KeyError:
            # Was not there
            pass

        # Check if that concept was in controlled_hosts
        try:
            _ = concept_mapping['controlled_hosts'][prev_concepts_host_idx]
            # Rename it from controlled hosts
            if type(concept_mapping['controlled_hosts'][prev_concepts_host_idx]) is set: 
                concept_mapping['controlled_hosts'][prev_concepts_host_idx].discard(ip)
                if len(concept_mapping['controlled_hosts'][prev_concepts_host_idx]) == 0:
                    del concept_mapping['controlled_hosts'][prev_concepts_host_idx] 
            else:
                del concept_mapping['controlled_hosts'][prev_concepts_host_idx]
            concept_mapping['controlled_hosts'][new_concepts_host_idx] = ip
        except KeyError:
            # Was not there
            pass

        # Check if that concept was in known_data
        try:
            _ = concept_mapping['known_data'][prev_concepts_host_idx]
            # Remove it from known data
            if type(concept_mapping['known_data'][prev_concepts_host_idx]) is set: 
                concept_mapping['known_data'][prev_concepts_host_idx].discard(ip)
                if len(concept_mapping['known_data'][prev_concepts_host_idx]) == 0:
                    del concept_mapping['known_data'][prev_concepts_host_idx] 
            else:
                del concept_mapping['known_data'][prev_concepts_host_idx]
            concept_mapping['known_data'][new_concepts_host_idx] = ip
        except KeyError:
            # Was not there
            pass

        # Check if that concept was in known_blocks
        try:
            _ = concept_mapping['known_blocks'][prev_concepts_host_idx]
            # Remove it from known blocks
            if type(concept_mapping['known_blocks'][prev_concepts_host_idx]) is set: 
                concept_mapping['known_blocks'][prev_concepts_host_idx].discard(ip)
                if len(concept_mapping['known_blocks'][prev_concepts_host_idx]) == 0:
                    del concept_mapping['known_blocks'][prev_concepts_host_idx] 
            else:
                del concept_mapping['known_blocks'][prev_concepts_host_idx]
            concept_mapping['known_blocks'][new_concepts_host_idx] = ip
        except KeyError:
            # Was not there
            pass



    ##########################
    # Convert Data
    # state.known_data: dict. {'ip': Data object}
    #   Data - Object
    #       data.ownwer
    #       data.id
    for ip, data in state.known_data.items():
        concept_host_idx = ip_to_concept[ip]
        concept_mapping['known_data'][concept_host_idx] = data


    ##########################
    # Convert Networks

    for network in state.known_networks:
        # We dont want to use external networks
        if not network.is_private():
            continue

        # Find if we know hosts in this network, if we do, assign new name
        number_hosts = 0
        for known_host in state.known_hosts:
            if ipaddress.IPv4Address(known_host.ip) in ipaddress.IPv4Network(f'{network.ip}/{network.mask}'):
                # There are hosts we know in this network
                number_hosts += 1

        # Add the unique network
        new_net = Network('net_' + str(network_concept_counter) + '_' + str(number_hosts) + 'hosts', 24)
        network_concept_counter += 1

        # Now store in ip_to_concept
        ip_to_concept[network] = new_net
        # And store in concept_mapping
        concept_mapping['known_networks'][new_net] = network

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
        concept_observation: The tuple that inside as an observation and the concept mapping
        use_controlled_hosts: If True, search in controlled_hosts, otherwise in known_hosts

    Returns:
        The corresponding IP address
    """
    host_mapping_set = concept_observation.concept_mapping['controlled_hosts'] if use_controlled_hosts else concept_observation.concept_mapping['known_hosts']

    # Check if the target host concept exists in the mapping
    if target_host_concept in host_mapping_set:
        mapped_value = host_mapping_set[target_host_concept]
        return mapped_value

def _convert_source_host_concept_to_ip(src_host_concept, concept_observation):
    """
    Helper function to convert source host concept to IP.
    Source hosts are always from controlled_hosts.

    Args:
        src_host_concept: The concept host to convert
        concept_observation: The tuple that inside as an observation and the concept mapping

    Returns:
        The corresponding IP address
    """
    controlled_hosts_dict = concept_observation.concept_mapping['controlled_hosts']

    # Check if the source host concept exists in the mapping
    if src_host_concept in controlled_hosts_dict:
        return controlled_hosts_dict[src_host_concept]

def _convert_network_concept_to_ip(target_net_concept, concept_observation):
    """
    Helper function to convert network concept to actual network.

    Args:
        target_net_concept: The concept network to convert
        concept_observation: The tuple that inside as an observation and the concept mapping

    Returns:
        The corresponding network object
    """
    networks_dict = concept_observation.concept_mapping['known_networks']

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

    return target_net_concept


def convert_concepts_to_actions(action, observation, concept_logger=None):
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