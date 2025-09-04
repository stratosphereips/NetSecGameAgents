import argparse
import pickle
from colorama import Fore, init

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
    # Get max valid state id
    max_state = len(states) - 1
    
    # Validate state range
    if args.state_id > max_state:
        print(f"Error: state_id {args.state_id} is out of range. Max state is {max_state}")
        return
    
    last_state = min(args.last_state_id, max_state) if args.last_state_id > 0 else args.state_id
    
    print(f"Showing states from {args.state_id} to {last_state} (max available: {max_state})")

    for state in range(args.state_id, last_state + 1):
        try:
            print(f'\n-------------------------------------')
            print(f'State {state}: {list(states.items())[state]}')
            filtered_items = {key: value for key, value in q_values.items() if key[0] == state}

            sorted_items = dict(sorted(filtered_items.items(), key=lambda item: item[1], reverse=True))

            # Identify the maximum value 
            max_value = next(iter(sorted_items.values()), None)

            for index, (key, value) in enumerate(sorted_items.items()):
                if value == max_value:
                    print(Fore.RED + f'\t{key} -> {value}' + Fore.RESET)
                else:
                    if not args.only_top:
                        print(Fore.GREEN + f'\t{key} -> {value}' + Fore.RESET)
        except IndexError:
            print(f"Error: Could not access state {state}")
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser('You can train the agent, or test it. \n Test is also to use the agent. \n During training and testing the performance is logged.')
    parser.add_argument("--file", help="Q-table file to load", default="q_agent_marl.pickle", required=False, type=str)
    parser.add_argument("--state_id", help="ID of the state to print", default=0, required=False, type=int)
    parser.add_argument("--last_state_id", help="Last ID of the state to print", default=0, required=False, type=int)
    parser.add_argument("--only_top", help="Print only the top action, the one to be taken if greedy", default=False, required=False, type=bool)
    args = parser.parse_args()

    # For the colorama
    init(strip=False)  # Changed from autoreset=True

    load_q_table()

    show_q_table()