import argparse
from math import inf
import pickle
from colorama import Fore, Style, init

#q_values = {}
#states = {}

def load_q_table():
    global q_values
    global states
    print(f'Loading file {args.file}')
    with open(args.file, "rb") as f:
        data = pickle.load(f)
        q_values = data["q_table"]
        states = data["state_mapping"]
    print(f'Len of qtable: {len(q_values)}')

def show_q_table():
    """
    Show details about a state in the qtable
    """
    print(f'State: {list(states.items())[args.state_id]}')

    filtered_items = {key: value for key, value in q_values.items() if key[0] == args.state_id}

    sorted_items = dict(sorted(filtered_items.items(), key=lambda item: item[1], reverse=True))

    # Identify the maximum value
    max_value = next(iter(sorted_items.values()), None)

    for index, (key, value) in enumerate(sorted_items.items()):
        if value == max_value:
            print(Fore.RED + f'{key} -> {value}')
        else:
            print(Fore.GREEN + f'{key} -> {value}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('You can train the agent, or test it. \n Test is also to use the agent. \n During training and testing the performance is logged.')
    parser.add_argument("--file", help="Q-table file to load", default="q_agent_marl.pickle", required=False, type=str)
    parser.add_argument("--state_id", help="ID of the state to print", default=0, required=False, type=int)
    args = parser.parse_args()

    init(autoreset=True)

    load_q_table()

    show_q_table()