import argparse
import logging
import time
from os import path
from AIDojoCoordinator.game_components import Action, ActionType, IP, Network, Service
from NetSecGameAgents.agents.base_agent import BaseAgent

def winning_strat_cyst(host, port, delay=1):
    agent1 = BaseAgent(host, port, role="Attacker")
    obs1 = agent1.register()
    time.sleep(delay)
    print(obs1)
    print("----------------------------")
    # # network scan
    obs1 = agent1.make_step(Action(
        ActionType.ScanNetwork,
        parameters={
            "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
            "target_network":Network("192.168.1.0", 24) 
            }
        )
    )
    print(obs1)
    time.sleep(delay)
    print("----------------------------")
    obs1 = agent1.make_step(Action(
        ActionType.FindServices,
        parameters={
            "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
            "target_host": IP("192.168.1.10")
            }
        )
    )
    print(obs1)
    print("----------------------------")
    time.sleep(delay)
    obs1 = agent1.make_step(Action(
        ActionType.ExploitService,
        parameters={
            "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
            "target_host": IP("192.168.1.10"),
            "target_service": Service(name='ssh', type='unknown', version='5.1.4', is_local=True)
            }
        )
    )
    print(obs1)
    print("----------------------------")
    time.sleep(delay)
    obs1 = agent1.make_step(Action(
        ActionType.FindData,
        parameters={
            "source_host": IP("192.168.4.10"),
            "target_host": IP("192.168.1.10"),
            }
        )
    )
    print(obs1)
    print("----------------------------")
    time.sleep(delay)
    obs1 = agent1.make_step(Action(
        ActionType.FindData,
        parameters={
            "source_host": IP("192.168.1.10"),
            "target_host": IP("192.168.1.10"),
            }
        )
    )
    print(obs1)
    print("----------------------------")
    time.sleep(delay)
    obs1 = agent1.make_step(Action(
        ActionType.ExfiltrateData,
        parameters={
            "source_host": IP("192.168.1.10"),
            "target_host": IP("192.168.4.10"),
            "data":list(obs1.state.known_data[IP("192.168.1.10")])[0]
            }
        )
    )
    print(obs1)
    print("----------------------------")
    obs_reset = agent1.make_step(Action(action_type=ActionType.ResetGame, parameters={"request_trajectory":True}))
    print(obs_reset)

def winning_strat(host, port, delay=10):
    agent1 = BaseAgent(host, port, role="Attacker")
    obs1 = agent1.register()
    time.sleep(delay)
    print(obs1)
    print("----------------------------")
    # # network scan
    obs1 = agent1.make_step(Action(
        ActionType.ScanNetwork,
        parameters={
            "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
            "target_network":Network("192.168.1.0", 24) 
            }
        )
    )
    print(obs1)
    time.sleep(delay)
    print("----------------------------")
    obs1 = agent1.make_step(Action(
        ActionType.FindServices,
        parameters={
            "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
            "target_host": IP("192.168.1.2")
            }
        )
    )
    print(obs1)
    print("----------------------------")
    time.sleep(delay)
    obs1 = agent1.make_step(Action(
        ActionType.ExploitService,
        parameters={
            "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
            "target_host": IP("192.168.1.2"),
            "target_service": list(obs1.state.known_services[IP("192.168.1.2")])[0]
            }
        )
    )
    print(obs1)
    print("----------------------------")
    time.sleep(delay)
    obs1 = agent1.make_step(Action(
        ActionType.FindData,
        parameters={
            "source_host": list(filter(lambda x: x !=  IP("213.47.23.195"), obs1.state.controlled_hosts))[0],
            "target_host": IP("192.168.1.2"),
            }
        )
    )
    print(obs1)
    print("----------------------------")
    time.sleep(delay)
    obs1 = agent1.make_step(Action(
        ActionType.ExfiltrateData,
        parameters={
            "source_host": IP("192.168.1.2"),
            "target_host": IP("213.47.23.195"),
            "data":list(obs1.state.known_data[IP("192.168.1.2")])[0]
            }
        )
    )
    print(obs1)
    print("----------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    parser.add_argument("--delay", help="Delay between actions (in seconds)", default=2, type=int, action='store', required=False)
    parser.add_argument("--mode", help="Mode (default=nsg)", type=str,default="nsg", action='store', required=False)
    
    args = parser.parse_args()
    log_filename = path.dirname(path.abspath(__file__)) + '/base_agent.log'
    logging.basicConfig(filename="scripted_attacker.log", filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s',  datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    if args.mode.lower() == "cyst":
        winning_strat_cyst(args.host, args.port, args.delay)
    else:
        winning_strat(args.host, args.port, args.delay)
