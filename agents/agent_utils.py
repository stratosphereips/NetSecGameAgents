"""
Collection of functions which are intended for agents to share.
Functionality which is specific for a certain type of agent should be implementented
directly in the agent class. 

author: Ondrej Lukas - ondrej.lukas@aic.fel.cvut.cz
"""
from collections import namedtuple
import random
import ipaddress
import warnings
from netsecgame import Action, ActionType, GameState, Observation, Network
from netsecgame.game_components import AgentStatus
from netsecgame import generate_valid_actions as _generate_valid_actions_core
from netsecgame import state_as_ordered_string as _state_as_ordered_string_core

def generate_valid_actions_concepts(
    state: GameState,
    action_history: set,
    include_blocks=False,
    *,
    filter_scan_network=True,
    filter_find_services=True,
    filter_exploit_service=True,
    filter_find_data=True,
    filter_exfiltrate_data=True,
    allow_repeated_actions=False,
    single_source=False,
    allow_repeated_network_scans=False,
    allow_service_rescans=False,
    include_local_services=False,
    allow_exploit_controlled_hosts=False,
    allow_find_data_rescans=False,
    prohibit_find_data_self_targeting=False,
    include_logfile_exfiltration=False,
    allow_duplicate_data_exfiltration=False,
    exfiltrate_to_external_only=False,
    ignore_firewall=False,
)->list:
    """Generate valid conceptual actions for the attacker.

    Every ablation parameter is opt-in. Calling this function with only
    ``state`` and ``action_history`` preserves the original behavior.

    Args:
        state: Current conceptual game state.
        action_history: Conceptual actions already selected in this episode.
        include_blocks: Generate BlockIP actions.
        filter_scan_network: Apply the conceptual internal-network and
            internal-source restrictions to ScanNetwork. When false, generate
            every controlled-source and known-network combination.
        filter_find_services: Apply the conceptual internal-host and
            already-known-services restrictions to FindServices.
        filter_exploit_service: Apply the conceptual internal-host,
            non-local-service, uncontrolled-target, and non-self restrictions
            to ExploitService.
        filter_find_data: Apply the conceptual internal-host and unknown-data
            restrictions to FindData.
        filter_exfiltrate_data: Apply the conceptual internal-source, logfile,
            and duplicate-at-destination restrictions to ExfiltrateData.
        allow_repeated_actions: Ignore history for all action families.
        single_source: Use one deterministic internal controlled source for
            ScanNetwork, FindServices, ExploitService, and FindData.
            ExfiltrateData sources remain hosts containing the selected data.
        allow_repeated_network_scans: Ignore history for ScanNetwork only.
        allow_service_rescans: Scan hosts whose services are already known.
        include_local_services: Generate exploits for local services.
        allow_exploit_controlled_hosts: Exploit already controlled hosts.
        allow_find_data_rescans: Search hosts whose data is already known.
        prohibit_find_data_self_targeting: Require a different source and
            target for FindData.
        include_logfile_exfiltration: Generate exfiltration for logfile data.
        allow_duplicate_data_exfiltration: Exfiltrate data to a destination
            that already contains the same data id.
        exfiltrate_to_external_only: Restrict destinations to controlled
            conceptual hosts containing ``external``.
        ignore_firewall: Do not prune actions using known firewall blocks.
    """

    def is_fw_blocked(source_host, target_host) -> bool:
        if ignore_firewall:
            return False
        try:
            return target_host in state.known_blocks[source_host]
        except KeyError:
            return False

    def is_internal_concept(host) -> bool:
        return type(host) is str and 'external' not in host

    def action_is_allowed(action) -> bool:
        if allow_repeated_actions:
            return True
        if (
            action.action_type == ActionType.ScanNetwork
            and allow_repeated_network_scans
        ):
            return True
        return action not in action_history

    valid_actions = set()

    # Deterministic ordering keeps seeded ablation runs reproducible.
    controlled_hosts = sorted(state.controlled_hosts, key=str)
    known_hosts = sorted(state.known_hosts, key=str)
    known_networks = sorted(state.known_networks, key=str)
    source_hosts = controlled_hosts
    if single_source:
        internal_sources = [
            source_host
            for source_host in controlled_hosts
            if is_internal_concept(source_host)
        ]
        source_hosts = internal_sources[:1] or controlled_hosts[:1]

    for source_host in source_hosts:
        for network in known_networks:
            if (
                not filter_scan_network
                or (
                    type(network.ip) is str
                    and 'external' not in network.ip
                    and is_internal_concept(source_host)
                )
            ):
                action = Action(
                    ActionType.ScanNetwork,
                    parameters={
                        "target_network": network,
                        "source_host": source_host,
                    },
                )
                if action_is_allowed(action):
                    valid_actions.add(action)

    for source_host in source_hosts:
        for target_host in known_hosts:
            if (
                not is_fw_blocked(source_host, target_host)
                and not is_fw_blocked(target_host, source_host)
                and (
                    not filter_find_services
                    or (
                        is_internal_concept(target_host)
                        and is_internal_concept(source_host)
                        and (
                            allow_service_rescans
                            or target_host not in state.known_services
                        )
                    )
                )
            ):
                action = Action(
                    ActionType.FindServices,
                    parameters={
                        "target_host": target_host,
                        "source_host": source_host,
                    },
                )
                if action_is_allowed(action):
                    valid_actions.add(action)

    for source_host in source_hosts:
        for target_host, service_list in sorted(
            state.known_services.items(),
            key=lambda item: str(item[0]),
        ):
            if (
                not is_fw_blocked(source_host, target_host)
                and not is_fw_blocked(target_host, source_host)
                and (
                    not filter_exploit_service
                    or (
                        is_internal_concept(target_host)
                        and is_internal_concept(source_host)
                        and (
                            allow_exploit_controlled_hosts
                            or target_host not in state.controlled_hosts
                        )
                        and target_host != source_host
                    )
                )
            ):
                for service in sorted(
                    service_list,
                    key=lambda item: getattr(item, "name", ""),
                ):
                    if (
                        not filter_exploit_service
                        or include_local_services
                        or not service.is_local
                    ):
                        action = Action(
                            ActionType.ExploitService,
                            parameters={
                                "target_host": target_host,
                                "target_service": service,
                                "source_host": source_host,
                            },
                        )
                        if action_is_allowed(action):
                            valid_actions.add(action)

    for source_host in source_hosts:
        for target_host in controlled_hosts:
            if (
                not is_fw_blocked(source_host, target_host)
                and not is_fw_blocked(target_host, source_host)
                and (
                    not filter_find_data
                    or (
                        is_internal_concept(target_host)
                        and is_internal_concept(source_host)
                        and (
                            allow_find_data_rescans
                            or target_host not in state.known_data
                        )
                    )
                )
                and (
                    not prohibit_find_data_self_targeting
                    or target_host != source_host
                )
            ):
                action = Action(
                    ActionType.FindData,
                    parameters={
                        "target_host": target_host,
                        "source_host": source_host,
                    },
                )
                if action_is_allowed(action):
                    valid_actions.add(action)

    for source_host, data_list in sorted(
        state.known_data.items(),
        key=lambda item: str(item[0]),
    ):
        for target_host in controlled_hosts:
            if (
                target_host in state.controlled_hosts
                and not is_fw_blocked(source_host, target_host)
                and target_host != source_host
                and data_list is not None
                and (
                    not filter_exfiltrate_data
                    or is_internal_concept(source_host)
                )
                and (
                    not exfiltrate_to_external_only
                    or (
                        type(target_host) is str
                        and 'external' in target_host
                    )
                )
            ):
                for data in sorted(
                    data_list,
                    key=lambda item: (
                        getattr(item, "owner", ""),
                        getattr(item, "id", ""),
                    ),
                ):
                    if (
                        filter_exfiltrate_data
                        and data.id == 'logfile'
                        and not include_logfile_exfiltration
                    ):
                        continue

                    data_at_target = state.known_data.get(target_host, set())
                    data_was_not_exfiltrated_before = not any(
                        exfiltrated_data.id == data.id
                        for exfiltrated_data in data_at_target
                    )
                    action = Action(
                        ActionType.ExfiltrateData,
                        parameters={
                            "target_host": target_host,
                            "source_host": source_host,
                            "data": data,
                        },
                    )
                    if (
                        (
                            not filter_exfiltrate_data
                            or allow_duplicate_data_exfiltration
                            or data_was_not_exfiltrated_before
                        )
                        and action_is_allowed(action)
                    ):
                        valid_actions.add(action)

    if include_blocks:
        for source_host in controlled_hosts:
            for target_host in controlled_hosts:
                if (
                    is_internal_concept(source_host)
                    and is_internal_concept(target_host)
                    and not is_fw_blocked(source_host, target_host)
                ):
                    for blocked_host in known_hosts:
                        if blocked_host == source_host:
                            continue
                        action = Action(
                            ActionType.BlockIP,
                            {
                                "target_host": target_host,
                                "source_host": source_host,
                                "blocked_host": blocked_host,
                            },
                        )
                        if action_is_allowed(action):
                            valid_actions.add(action)

    return sorted(valid_actions, key=str)

def generate_valid_actions(state: GameState, include_blocks=False) -> list:
    warnings.warn(
        "Importing generate_valid_actions from 'NetSecGameAgents.agents.agent_utils' is deprecated. "
        "Please import directly from 'netsecgame' as follows: 'from netsecgame import generate_valid_actions'.",
        DeprecationWarning,
        stacklevel=2
    )
    return _generate_valid_actions_core(state, include_blocks)    

def state_as_ordered_string(state: GameState) -> str:
    warnings.warn(
        "Importing state_as_ordered_string from 'NetSecGameAgents.agents.agent_utils' is deprecated. "
        "Please import directly from 'netsecgame' as follows: 'from netsecgame import state_as_ordered_string'.",
        DeprecationWarning,
        stacklevel=2
    )
    return _state_as_ordered_string_core(state)

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
    # Iterate in deterministic order over real hosts
    for host in sorted(state.known_hosts):
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

    for ip in sorted(state.known_services.keys()):
        services = state.known_services[ip]
        # Get port numbers
        port_numbers = ''
        for service in sorted(services, key=lambda s: s.name):
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
    for ip, data in sorted(state.known_data.items(), key=lambda kv: str(kv[0])):
        concept_host_idx = ip_to_concept[ip]
        concept_mapping['known_data'][concept_host_idx] = data


    ##########################
    # Convert Networks

    for network in sorted(state.known_networks):
        # We dont want to use external networks
        if not network.is_private():
            continue

        # Find if we know hosts in this network, if we do, assign new name
        number_hosts = 0
        for known_host in sorted(state.known_hosts):
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

def _convert_network_concept_to_ip(target_net_concept, concept_observation, rng=None):
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
        # Keep randomness via choice, but apply it over a deterministically
        # ordered list so that with a fixed seed the behaviour is reproducible.
        if isinstance(mapped_value, set):
            choices = sorted(mapped_value, key=str)
            if rng is not None:
                return rng.choice(choices)
            return random.choice(choices)
        return mapped_value

    return target_net_concept


def convert_concepts_to_actions(action, observation, concept_logger=None, rng=None):
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
        new_target_network = _convert_network_concept_to_ip(action.parameters['target_network'], observation, rng=rng)
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
