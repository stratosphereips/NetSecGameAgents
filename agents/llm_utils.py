"""
Collection of functions used by LLM agents.

author: Maria Rigaki - maria.rigaki@aic.fel.cvut.cz
"""
from AIDojoCoordinator.game_components import (
    ActionType,
    Action,
    IP,
    Data,
    Network,
    Service,
    GameState,
)

local_services = ["can_attack_start_here"]


def create_status_from_state(state: GameState) -> str:
    """Create a status prompt using the current state and the sae memories."""
    contr_hosts = [host.ip for host in state.controlled_hosts]
    known_hosts = [
        str(host) for host in state.known_hosts if host.ip not in contr_hosts
    ]
    known_nets = [str(net) for net in list(state.known_networks)]

    prompt = "Current status:\n"
    prompt += f"Controlled hosts are {' and '.join(contr_hosts)}\n"
    # logger.info("Controlled hosts are %s", " and ".join(contr_hosts))

    prompt += f"Known networks are {' and '.join(known_nets)}\n"
    # logger.info("Known networks are %s", " and ".join(known_nets))
    prompt += f"Known hosts are {' and '.join(known_hosts)}\n"
    # logger.info("Known hosts are %s", " and ".join(known_hosts))

    if len(state.known_services.keys()) == 0:
        prompt += "Known services are none\n"
        # logger.info(f"Known services: None")
    for ip_service in state.known_services:
        services = []
        if len(list(state.known_services[ip_service])) > 0:
            for serv in state.known_services[ip_service]:
                if serv.name not in local_services:
                    services.append(serv.name)
            if len(services) > 0:
                serv_str = ""
                for serv in services:
                    serv_str += serv + " and "
                prompt += f"Known services for host {ip_service} are {serv_str}\n"
                # logger.info(f"Known services {ip_service, services}")
            else:
                prompt += "Known services are none\n"
                # logger.info(f"Known services: None")

    if len(state.known_data.keys()) == 0:
        prompt += "Known data are none\n"
        # logger.info(f"Known data: None")
    for ip_data in state.known_data:
        if len(state.known_data[ip_data]) > 0:
            host_data = ""
            for known_data in list(state.known_data[ip_data]):
                host_data += f"({known_data.owner}, {known_data.id}) and "
            prompt += f"Known data for host {ip_data} are {host_data}\n"
            # logger.info(f"Known data: {ip_data, state.known_data[ip_data]}")

    return prompt


def validate_action_in_state(llm_response: dict, state: GameState) -> bool:
    """Check the LLM response and validate it against the current state."""
    contr_hosts = [str(host) for host in state.controlled_hosts]
    known_hosts = [
        str(host) for host in state.known_hosts if host.ip not in contr_hosts
    ]
    known_nets = [str(net) for net in list(state.known_networks)]

    valid = False
    try:
        action_str = llm_response["action"]
        action_params = llm_response["parameters"]
        if isinstance(action_params, str):
            action_params = eval(action_params)
        match action_str:
            case "ScanNetwork":
                if action_params["target_network"] in known_nets:
                    valid = True
            case "ScanServices" | "FindServices":
                if (
                    action_params["target_host"] in known_hosts
                    or action_params["target_host"] in contr_hosts
                ):
                    valid = True
            case "ExploitService":
                ip_addr = action_params["target_host"]
                if ip_addr in known_hosts:
                    valid = True
                    # for service in state.known_services[IP(ip_addr)]:
                    #     if service.name == action_params["target_service"]:
                    #         valid = True
            case "FindData":
                if action_params["target_host"] in contr_hosts:
                    valid = True
            case "ExfiltrateData":
                for ip_data in state.known_data:
                    ip_addr = action_params["source_host"]
                    if ip_data == IP(ip_addr) and ip_addr in contr_hosts:
                        valid = True
            case _:
                valid = False
        return valid
    except:
        # logger.info("Exception during validation of %s", llm_response)
        return False


def create_action_from_response(llm_response: dict, state: GameState) -> tuple:
    """Build the action object from the llm response"""
    try:
        # Validate action based on current states
        valid = validate_action_in_state(llm_response, state)
        action = None
        action_str = llm_response["action"]
        action_params = llm_response["parameters"]

        if valid:
            match action_str:
                case "ScanNetwork":
                    target_net, mask = action_params["target_network"].split("/")
                    src_host = action_params["source_host"]
                    action = Action(
                        ActionType.ScanNetwork,
                        {
                            "target_network": Network(target_net, int(mask)),
                            "source_host": IP(src_host),
                        },
                    )
                case "ScanServices" | "FindServices":
                    src_host = action_params["source_host"]
                    action = Action(
                        ActionType.FindServices,
                        {
                            "target_host": IP(action_params["target_host"]),
                            "source_host": IP(src_host),
                        },
                    )
                case "ExploitService":
                    target_ip = action_params["target_host"]
                    target_service = action_params["target_service"]
                    src_host = action_params["source_host"]
                    if len(list(state.known_services[IP(target_ip)])) > 0:
                        for serv in state.known_services[IP(target_ip)]:
                            if serv.name == target_service.lower():
                                parameters = {
                                    "target_host": IP(target_ip),
                                    "target_service": Service(
                                        serv.name,
                                        serv.type,
                                        serv.version,
                                        serv.is_local,
                                    ),
                                    "source_host": IP(src_host),
                                }
                                action = Action(ActionType.ExploitService, parameters)
                                break
                    else:
                        action = None
                case "FindData":
                    src_host = action_params["source_host"]
                    action = Action(
                        ActionType.FindData,
                        {
                            "target_host": IP(action_params["target_host"]),
                            "source_host": IP(src_host),
                        },
                    )
                case "ExfiltrateData":
                    try:
                        # data_owner, data_id = action_params["data"]
                        data_owner = action_params["data"]["owner"]
                        data_id = action_params["data"]["id"]
                    except:
                        action_data = eval(action_params["data"])
                        data_owner = action_data["owner"]
                        data_id = action_data["id"]

                    action = Action(
                        ActionType.ExfiltrateData,
                        {
                            "target_host": IP(action_params["target_host"]),
                            "data": Data(data_owner, data_id),
                            "source_host": IP(action_params["source_host"]),
                        },
                    )
                case _:
                    return False, action

    except SyntaxError:
        valid = False

    return valid, action
