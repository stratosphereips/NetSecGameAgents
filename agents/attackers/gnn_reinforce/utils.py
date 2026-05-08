from netsecgame import Observation


def filter_log_files_from_state(observation: Observation) -> Observation:
    """Removes verbose 'logfile' entities from the observation state."""
    for host in list(observation.state.known_data.keys()):
        data_list = observation.state.known_data[host]
        filtered_data = [data for data in data_list if data.id != "logfile"]
        if len(filtered_data) > 0:
            observation.state.known_data[host] = filtered_data
        else:
            del observation.state.known_data[host]
    return observation
