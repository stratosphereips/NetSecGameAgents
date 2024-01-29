import sys
import os
sys.path.append(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))
#with the path fixed, we can import now
from env.game_components import Action, ActionType, GameState, Observation


def generate_valid_actions(state: GameState)->list:
        valid_actions = set()
        for src_host in state.controlled_hosts:
            #Network Scans
            for network in state.known_networks:
                # TODO ADD neighbouring networks
                valid_actions.add(Action(ActionType.ScanNetwork, params={"target_network": network, "source_host": src_host,}))
            # Service Scans
            for host in state.known_hosts:
                valid_actions.add(Action(ActionType.FindServices, params={"target_host": host, "source_host": src_host,}))
            # Service Exploits
            for host, service_list in state.known_services.items():
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