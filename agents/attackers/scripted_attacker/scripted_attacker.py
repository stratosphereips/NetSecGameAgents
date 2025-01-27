import sys
import argparse
import logging
from os import path
import time

sys.path.append(path.dirname(path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__) ))))))
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__) ))))

#with the path fixed, we can import now
from AIDojoCoordinator.game_components import Action, ActionType, IP, Network
from base_agent import BaseAgent

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
    parser.add_argument("--delay", help="Delay between actions (in seconds)", default=10, type=int, action='store', required=False)
    
    args = parser.parse_args()
    log_filename = path.dirname(path.abspath(__file__)) + '/base_agent.log'
    logging.basicConfig(filename="scripted_attacker.log", filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s',  datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    winning_strat(args.host, args.port, args.delay)