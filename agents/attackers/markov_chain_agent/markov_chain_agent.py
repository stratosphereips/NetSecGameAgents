import sys
from os import path
# This is used so the agent can see the environment and game components
sys.path.append(path.dirname(path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__) ) ) ))))
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__) ))))

import logging
import os
from random import choice
import argparse
import numpy as np
import math
import yaml

import pandas as pd 
import copy
import json
import csv
import time
import random
from env.worlds.network_security_game import NetworkSecurityEnvironment
from os import path



# with the path fixed, we can import now
from env.game_components import Action, Observation, GameState, ActionType
from base_agent import BaseAgent
from agent_utils import generate_valid_actions
from datetime import datetime

import json

class GeneticAgent(BaseAgent):

    def __init__(self, host, port,role, episodes) -> None:
        super().__init__(host, port, role)
        np.set_printoptions(suppress=True, precision=6)
        self.parsed_solutions = []
        self.episodes = episodes

    def append_to_parsed_solutions(self, individual):
        self.parsed_solutions.append(individual)

    def parse_action(action):
        return {
            "type": action.action_type.name,
            "params": action.params
        }
    

    def normalize_probabilities(transitions):
            """Function to normalize transition probabilities to ensure they sum up to 1."""
            normalized_transitions = {}
            
            for action_type, probs in transitions.items():
                total = sum(probs)
                if total != 1:
                    # Normalize the probabilities by dividing each by the total sum
                    normalized_transitions[action_type] = [prob / total for prob in probs]
                else:
                    normalized_transitions[action_type] = probs  # Already normalized

            return normalized_transitions
       
       
    # Load JSON data from a file
    with open("transition_probabilities.json", "r") as file:
        transitions_data = json.load(file)

    # Initialize an empty dictionary to store the transitions
    transitions = {}

    # Mapping from string keys to ActionType enum members
    action_mapping = {
        "ScanNetwork": ActionType.ScanNetwork,
        "FindServices": ActionType.FindServices,
        "ExploitService": ActionType.ExploitService,
        "FindData": ActionType.FindData,
        "ExfiltrateData": ActionType.ExfiltrateData,
    }

    # Loop through each action's transition probabilities
    for action_data in transitions_data["transition_probabilities"]:
        action = action_data["Action"]

        # Extract the probabilities and store them in the correct order
        probabilities = [
            action_data["ScanNetwork"],
            action_data["FindServices"],
            action_data["ExploitService"],
            action_data["FindData"],
            action_data["ExfiltrateData"]
        ]

        # Assign the probabilities list to the correct action key in the transitions dictionary
        if action == "Initial Action":
            transitions["Initial"] = probabilities
        else:
            # Use the ActionType mapping for other actions
            transitions[action_mapping[action]] = probabilities



    transitions = normalize_probabilities(transitions)


    def generate_valid_actions_separated(self,state: GameState)->list:
        """Function that generates a list of all valid actions in a given state"""
        valid_scan_network = set()
        valid_find_services = set()
        valid_exploit_service = set()
        valid_find_data = set()
        valid_exfiltrate_data = set()


        for src_host in state.controlled_hosts:
            #Network Scans
            for network in state.known_networks:
                valid_scan_network.add(Action(ActionType.ScanNetwork, params={"target_network": network, "source_host": src_host,}))
            # Service Scans
            for host in state.known_hosts:
                valid_find_services.add(Action(ActionType.FindServices, params={"target_host": host, "source_host": src_host,}))
            # Service Exploits
            for host, service_list in state.known_services.items():
                for service in service_list:
                    valid_exploit_service.add(Action(ActionType.ExploitService, params={"target_host": host,"target_service": service,"source_host": src_host,}))
        # Data Scans
        for host in state.controlled_hosts:
            valid_find_data.add(Action(ActionType.FindData, params={"target_host": host, "source_host": host}))

        # Data Exfiltration
        for src_host, data_list in state.known_data.items():
            for data in data_list:
                for trg_host in state.controlled_hosts:
                    if trg_host != src_host:
                        valid_exfiltrate_data.add(Action(ActionType.ExfiltrateData, params={"target_host": trg_host, "source_host": src_host, "data": data}))


        return list(valid_scan_network), list(valid_find_services), list(valid_exploit_service), list(valid_find_data), list(valid_exfiltrate_data), 
  
    
    def select_action_markov_chain_agent(self, observation: Observation, lastActionType) -> Action:
        # Generate valid actions as a tuple of lists
        valid_actions = self.generate_valid_actions_separated(observation.state)
        # Transition probabilities

        

        # Set default lastActionType
        if lastActionType is None:
            lastActionType = "Initial"
        
        # Ensure the length of transitions matches the number of action categories
        if len(valid_actions) != len(self.transitions[lastActionType]):
            raise ValueError(f"Mismatch between valid action lists and transition probabilities: {len(valid_actions)} vs {len(self.transitions[lastActionType])}")

        # Step 1: Select which action type to pick from, based on probabilities
        actions_to_pick_from = 0
        selected_action_type_index = None
        selected_action = None
        while selected_action is None:
            selected_action_type_index = np.random.choice(len(valid_actions), p=self.transitions[lastActionType])
                
            selected_action_list = valid_actions[selected_action_type_index]
            actions_to_pick_from = len(selected_action_list)
            if actions_to_pick_from > 0:
                selected_action = np.random.choice(selected_action_list)
        return selected_action
    
    def play_game_markov_chain_agent(self, observation):
        """
        The main function for the gameplay. Handles agent registration and the main interaction loop.
        """
        num_steps = 0
        actions = []
        taken_actions = {}
        episodic_returns = []
        lastActionType = None
        while observation and not observation.end:
            num_steps += 1
            # Store returns in the episode
            episodic_returns.append(observation.reward)
            # Select the action randomly
            action = agent.select_action_markov_chain_agent(observation, lastActionType)
            lastActionType = action.type
            taken_actions[action] = True
            actions.append(action)
            
            observation = agent.make_step(action)

        
        return actions



    def play_game(self):
        """
        The main function for the gameplay. Handles agent registration and the main interaction loop.
        """
        DEFAULT_PATH_RESULTS = "./results"


        def analyze_solution(solution, observation):
            i = 0
            num_good_actions = 0
            num_boring_actions = 0
            num_bad_actions = 0


            solution_result = [[0,0] for _ in range(len(solution))]

            if observation is None:
                observation = agent.request_game_reset()

            current_state = observation.state
            while i < len(solution) and not observation.end:
                valid_actions = generate_valid_actions(current_state)

                solution_result[i][0] = solution[i]
                
                if solution[i] in valid_actions:
                    observation = agent.make_step(solution[i])

                new_state = observation.state

                
                if current_state != new_state:
                    num_good_actions += 1
                    solution_result[i][1] = 1
                else:
                    if solution[i] in valid_actions:
                        num_boring_actions += 1
                        solution_result[i][1] = 0

                    else:
                        num_bad_actions += 1
                        solution_result[i][1] = -1
                current_state = observation.state
                i += 1
            
            if "end_reason" in observation.info and observation.info["end_reason"] == "goal_reached":
                solution_result[i - 1][1] = 9

            parsed_solution = []
            i = 0
            while i < len(solution):
                parsed_solution.append([solution[i], solution_result[i][1]])
                
                i += 1
                if solution_result[i - 1][1] == 9:
                    break
            agent.append_to_parsed_solutions(parsed_solution)

            return 
        

        # Fix, this should be episodes like in random agent
        episodes = self.episodes

        PATH_RESULTS = DEFAULT_PATH_RESULTS

        solutions = [None] * int(episodes)
        for i in range(int(episodes)):
            solutions[i] = agent.play_game_markov_chain_agent(agent.request_game_reset())
            #print(" Solution ", i, " done")


        for solution in solutions:
            analyze_solution(solution, agent.request_game_reset())
        
        def save_solutions_json(agent, path):
            # Step 1: Load existing data or initialize an empty outer array
            if os.path.exists(path):
                with open(path, "r") as f:
                    try:
                        outer_array = json.load(f)
                    except json.JSONDecodeError:
                        outer_array = []  # Initialize if the file is empty or invalid
            else:
                outer_array = []

            # Step 2: Convert the parsed population to the required format
            parsed_solutions_run = []
            for solution in agent.parsed_solutions:
                solution_str_list = []
                for action_result in solution:
                    if isinstance(action_result, tuple) and len(action_result) == 2:
                        # Format each action and result pair as a single string
                        action_str = f"[Action {str(action_result[0])}, {str(action_result[1])}]"
                        solution_str_list.append(action_str)
                    else:
                        # Handle cases where the structure isn't as expected
                        solution_str_list.append(str(action_result))

                parsed_solutions_run.append(solution_str_list)

            # Step 3: Append this run's population to the outer array
            outer_array.append(parsed_solutions_run)

            # Step 4: Write the updated outer array back to the file
            with open(path, "w") as f:
                json.dump(outer_array, f, indent=4)

        # Usage
        save_solutions_json(agent, os.path.join(PATH_RESULTS, 'parsed_population.json'))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    parser.add_argument("--episodes", help="Sets number of episodes to play or evaluate", default=100, type=int) 
    args = parser.parse_args()

 
    agent = GeneticAgent(args.host, args.port,"Attacker", args.episodes)



    observation = agent.register()
    agent.play_game()
    agent._logger.info("Terminating interaction")
    agent.terminate_connection()
    