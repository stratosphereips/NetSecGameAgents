"""
Collection of functions which are intended for graph based agents to share.
Functionality which is specific for a certain type of agent should be implementented
directly in the agent class. 

author: Ondrej Lukas - ondrej.lukas@aic.fel.cvut.cz
"""
import ipaddress
import numpy as np
from AIDojoCoordinator.game_components import GameState, IP, Network, Data, Service

def state_as_graph(state:GameState) -> tuple:
    node_types = {
        "network":0,
        "known_host":1,
        "controlled_host":2,
        "service":3,
        "datapoint":4
    }
    graph_nodes = {}
    total_nodes = len(state.known_networks) + len(state.known_hosts) + sum([len(x) for x in state.known_services.values()]) + sum([len(x) for x in state.known_data.values()])
    edge_list = []
    node_features = np.zeros([total_nodes, len(node_types)],dtype=np.int32)
    
    # networks
    for net in state.known_networks:
        graph_nodes[net] = len(graph_nodes)
        node_features[graph_nodes[net]][node_types["network"]] = 1    
    # hosts
    for host in state.known_hosts:
        graph_nodes[host] = len(graph_nodes)
        # add to adjacency matrix if it is part of network
        ip_addr = ipaddress.ip_address(str(host))
        for net in state.known_networks:
            if ip_addr in ipaddress.ip_network(str(net), strict=False):
                edge_list.append((graph_nodes[net], graph_nodes[host]))
                edge_list.append((graph_nodes[host], graph_nodes[net]))
        if host in state.controlled_hosts:
            node_features[graph_nodes[host]][node_types["controlled_host"]] = 1   
        else:
            node_features[graph_nodes[host]][node_types["known_host"]] = 1
    # services
    for host, service_list in state.known_services.items():
        for service in service_list:
            graph_nodes[service] = len(graph_nodes)
            node_features[graph_nodes[service]][node_types["service"]] = 1
            edge_list.append((graph_nodes[host], graph_nodes[service]))
            edge_list.append((graph_nodes[service], graph_nodes[host]))
    # data
    for host, data_list in state.known_data.items():
        for data in data_list:
            graph_nodes[data] = len(graph_nodes)
            node_features[graph_nodes[data]][node_types["datapoint"]] = 1
            edge_list.append((graph_nodes[host], graph_nodes[data]))
            edge_list.append((graph_nodes[data], graph_nodes[host]))
    
    # make edges bidirectional
    return node_features, edge_list

if __name__ == '__main__':
    state = GameState(known_networks={Network("192.168.1.0", 24),Network("1.1.1.2", 24)},
                known_hosts={IP("192.168.1.2"), IP("192.168.1.3")}, controlled_hosts={IP("192.168.1.2")},
                known_services={IP("192.168.1.3"):{Service("service1", "public", "1.01", True)}},
                known_data={IP("192.168.1.3"):{Data("ChuckNorris", "data1"), Data("ChuckNorris", "data2")},
                            IP("192.168.1.2"):{Data("McGiver", "data2")}})
    X,A = state_as_graph(state)
    print(X)
    print(A)