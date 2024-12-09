import sys
from os import path
from pathlib import Path

# Get the current file's directory and resolve the project root
current_file = Path(__file__).resolve()
project_root = current_file.parent
while project_root.name and not (project_root / 'env').exists():
    project_root = project_root.parent

if not project_root.name:
    raise Exception("Could not find project root (directory containing 'env/')")

# Set up basePath as before
basePath = str(project_root)

# Add parent directory to sys.path (where base_agent.py and agent_utils.py are)
parent_dir = current_file.parent.parent
sys.path.append(str(parent_dir))

# Add project root to sys.path
sys.path.append(basePath)

import logging
import os
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
from random import choice
from env.worlds.network_security_game import NetworkSecurityEnvironment
from env.game_components import Action, Observation, GameState, ActionType
from NetSecGameAgents.agents.base_agent import BaseAgent
from NetSecGameAgents.agents.agent_utils import generate_valid_actions
from datetime import datetime


class GeneticAgent(BaseAgent):

    def __init__(self, host, port,role) -> None:
        super().__init__(host, port, role)
        np.set_printoptions(suppress=True, precision=6)
        self.parsed_population = []

    def append_to_parsed_population(self, individual):
        self.parsed_population.append(individual)

    def parse_action(action):
        return {
            "type": action.action_type.name,
            "params": action.params
        }

  
    def select_action_random_agent(self, observation: Observation) -> Action:
        # Select a random action from the valid actions, except block actions
        action = choice([a for a in generate_valid_actions(observation.state) if a.type != ActionType.BlockIP])
        #print(f"Selected action: {action}")
        return action
    
    def play_game_random_agent(self, observation):
        """
        The main function for the gameplay. Handles agent registration and the main interaction loop.
        """
        num_steps = 0
        actions = []
        taken_actions = {}
        episodic_returns = []
        while observation and not observation.end:
            num_steps += 1
            # Store returns in the episode
            episodic_returns.append(observation.reward)
            # Select the action randomly
            action = agent.select_action_random_agent(observation)
            taken_actions[action] = True
            actions.append(action)
            
            observation = agent.make_step(action)

        # select random actions to fill the rest of the list
        while len(actions) < 100:
            #print("Filling the rest of the list")
            valid_action = agent.select_action_random_agent(observation)
            actions.append(valid_action)
        
        return actions



    def play_game(self):
        """
        The main function for the gameplay. Handles agent registration and the main interaction loop.
        """

        DEFAULT_PATH_RESULTS = "./results"

        def choose_parents_tournament(population, goal, fitness_func, num_per_tournament=2, parents_should_differ=True):
            """ Tournament selection """
            from_population = population.copy()
            chosen = []
            for i in range(2):
                options = []
                for _ in range(num_per_tournament):
                    options.append(random.choice(from_population))
                chosen.append(max(options, key=lambda x:fitness_func(x,agent.request_game_reset(),goal)[0])) # add [0] because fitness_eval_v3 returns a tuple
                if i==0 and parents_should_differ:
                    from_population.remove(chosen[0])
            return chosen[0], chosen[1]
        
        def mutation_operator_by_parameter(individual, all_actions_by_type, mutation_prob):
            new_individual = []
            for i in range(len(individual)):
                if random.random() < mutation_prob:
                    action_type = individual[i].type
                    new_individual.append(random.choice(all_actions_by_type[str(action_type)]))
                else:
                    new_individual.append(individual[i])
            return new_individual


        def mutation_operator_by_action(individual, all_actions, mutation_prob):
            new_individual = []
            for i in range(len(individual)):
                if random.random() < mutation_prob:
                    new_individual.append(random.choice(all_actions))
                else: 
                    new_individual.append(individual[i])
            return new_individual


        def crossover_operator_Npoints(parent1, parent2, num_points, cross_prob):
            if random.random() < cross_prob:
                len_ind = len(parent1)
                cross_points = np.sort(np.random.choice(len_ind, num_points, replace=False))
                child1 = []
                child2 = []
                current_parent1 = parent1
                current_parent2 = parent2
                for i in range(len_ind):
                    child1.append(current_parent1[i])
                    child2.append(current_parent2[i])
                    if i in cross_points:
                        current_parent1 = parent2 if current_parent1 is parent1 else parent1
                        current_parent2 = parent1 if current_parent2 is parent2 else parent2
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()
            return child1, child2


        def crossover_operator_uniform(parent1, parent2, p_value, cross_prob):
            if random.random() < cross_prob:
                len_ind = len(parent1)
                child1 = []
                child2 = []
                for i in range(len_ind):
                    if random.random() < p_value:
                        child1.append(parent1[i])
                        child2.append(parent2[i])
                    else:
                        child1.append(parent2[i])
                        child2.append(parent1[i])
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()
            return child1, child2
        

        def fitness_eval_v02(individual, observation, is_final_generation, num_steps = 0):
            i = 0
            num_good_actions = 0
            num_boring_actions = 0
            num_bad_actions = 0
            reward = 0
            reward_goal = 0

            individual_result = [[0,0] for _ in range(len(individual))]

            if observation is None:
                observation = agent.request_game_reset()

            current_state = observation.state
            while i < len(individual) and not observation.end:
                valid_actions = generate_valid_actions(current_state)

                individual_result[i][0] = individual[i]
                
                if individual[i] in valid_actions:
                    observation = agent.make_step(individual[i])
                if num_steps is None:
                    num_steps = 0
                num_steps += 1 
                new_state = observation.state

                
                if current_state != new_state:
                    num_good_actions += 1
                    individual_result[i][1] = 1
                    action_type = individual[i].type
                    if action_type == ActionType.ScanNetwork:
                        reward = 10
                    elif action_type == ActionType.FindServices:
                        reward = 20
                    elif action_type == ActionType.ExploitService:
                        reward = 50
                    elif action_type == ActionType.FindData:
                        reward = 75
                    elif action_type == ActionType.ExfiltrateData:
                        reward = 75
                else:
                    if individual[i] in valid_actions:
                        reward += -10
                        num_boring_actions += 1
                        individual_result[i][1] = 0

                    else:
                        reward += -100
                        num_bad_actions += 1
                        individual_result[i][1] = -1
                current_state = observation.state
                i += 1
                #print(reward)
            

            if "end_reason" in observation.info and observation.info["end_reason"] == "goal_reached":
                individual_result[i - 1][1] = 9
                won = 1
            else:
                won = 0

            final_reward = reward + reward_goal
            div_aux = num_steps - num_good_actions + num_bad_actions


          
                
            #print(reward,reward_goal,num_steps,div_aux)
            if div_aux == 0:
                # i.e. when num_steps == num_good_actions and num_bad_actions == 0
                # if num_bad_actions > 0, then num_steps + num_bad_actions != num_good_actions because num_steps > num_good_actions
                div = num_steps
            else:
                div = div_aux

            if final_reward >= 0:
                return_reward = final_reward / div
            else:
                return_reward = final_reward 

            if won == 1:
                return_reward = 7500 + 100000/num_steps
            #print(return_reward, num_good_actions, num_boring_actions, num_bad_actions, num_steps)

            if is_final_generation is True:
                parsed_individual_result = []
                i = 0
                while i < len(individual):
                    parsed_individual_result.append([individual[i], individual_result[i][1]])
                    i += 1
                    if individual_result[i - 1][1] == 9:
                        agent.append_to_parsed_population(parsed_individual_result)
                        break

            return return_reward, num_good_actions, num_boring_actions, num_bad_actions, num_steps, won
        

        def steady_state_selection(parents, parents_scores, offspring, offspring_scores, num_replace):
            # parents
            best_indices_parents = np.argsort(parents_scores, axis=0)[:,0] # min to max fitness (higher is better)
            parents_sort = [parents[i] for i in best_indices_parents]
            # offspring
            best_indices_offspring = np.argsort(offspring_scores, axis=0)[:,0] # min to max fitness (higher is better)
            offspring_sort = [offspring[i] for i in best_indices_offspring]
            # new generation
            new_generation = parents_sort[num_replace:] + offspring_sort[population_size-num_replace:]
            return new_generation


        def get_all_actions_by_type(all_actions):
            all_actions_by_type = {}
            ScanNetwork_list=[]
            FindServices_list=[]
            ExploitService_list=[]
            FindData_list=[]
            ExfiltrateData_list=[]
            for i in range(len(all_actions)):
                if ActionType.ScanNetwork==all_actions[i].type:
                    ScanNetwork_list.append(all_actions[i])
                elif ActionType.FindServices==all_actions[i].type:
                    FindServices_list.append(all_actions[i])
                elif ActionType.ExploitService==all_actions[i].type:
                    ExploitService_list.append(all_actions[i])
                elif ActionType.FindData==all_actions[i].type:
                    FindData_list.append(all_actions[i])
                elif ActionType.ExfiltrateData==all_actions[i].type:
                    ExfiltrateData_list.append(all_actions[i])
                    #print("Action: ", all_actions[i])
            all_actions_by_type["ActionType.ScanNetwork"] = ScanNetwork_list
            all_actions_by_type["ActionType.FindServices"] = FindServices_list
            all_actions_by_type["ActionType.ExploitService"] = ExploitService_list
            all_actions_by_type["ActionType.FindData"] = FindData_list
            all_actions_by_type["ActionType.ExfiltrateData"] = ExfiltrateData_list
            return all_actions_by_type
        
        
        

        env = NetworkSecurityEnvironment(path.join(basePath, 'env', 'netsecenv_conf.yaml'))
        all_actions = env.get_all_actions()
        max_number_steps = env._max_steps

        all_actions_by_type = get_all_actions_by_type(all_actions)


        # GA parameters

        # Load the JSON configuration
        with open('config.json', 'r') as file:
            config = json.load(file)

        # Set variables based on the loaded configuration
        population_size = config["population_size"]
        num_generations = config["num_generations"]

        # Parents selection (tournament) parameters
        select_parents_with_replacement = config["replacement"]
        num_per_tournament = config["num_per_tournament"]

        # Crossover parameters
        Npoints = config["n_points"]
        if Npoints:
            num_points = config["num_points"]
        else:
            p_value = config["p_value"]
        cross_prob = config["cross_prob"]

        # Mutation parameters
        parameter_mutation = config["parameter_mutation"]
        mutation_prob = config["mutation_prob"]

        # Survivor selection parameters
        num_replace = config["num_replace"]

        initialization_with_random_agent = config["initialization_with_random_agent"]
        reward_threshold = config["reward_threshold"]

        PATH_RESULTS = DEFAULT_PATH_RESULTS



        # Initialize population
        population = [[random.choice(all_actions) for _ in range(max_number_steps)] for _ in range(population_size)]

        # the given percentage of the population is initialized with Random Agent behavior
        for i in range(int(population_size * initialization_with_random_agent)):
            population[i] = agent.play_game_random_agent(agent.request_game_reset())
            print("Random Agent behavior initialized: ", i)


        #print("Best initial fitness: ", max([fitness_eval_v02(individual, agent.request_game_reset(),False, 0)[0] for individual in population]))

        # Generations

        generation = 0
        best_score = -math.inf


        try:
            while (generation < num_generations) and (best_score < reward_threshold):
                print("Generation: ", generation)
                #print(generation)
                offspring = []
                #print("inic offspring")
                popu_crossover = population.copy()
                #print("copy population")
                parents_scores = np.array([fitness_eval_v02(individual, agent.request_game_reset(),False, 0) for individual in population])
                #print("parents_scores")
                index_best_score = np.argmax(parents_scores[:, 0])
                best_score_complete = parents_scores[index_best_score, :]
                best_score = best_score_complete[0]

                index_worst_score = np.argmin(parents_scores[:, 0])
                worst_score_complete = parents_scores[index_worst_score, :]

                #print("Amount of individuals: ", len(parents_scores))
                #print("Total good actions: ", np.sum(parents_scores[:, 1]))
                print("Best score complete: ", best_score_complete)
                print("Worst score complete: ", worst_score_complete)
                print("Average score complete: ", np.mean(parents_scores, axis=0))
                metrics_mean = np.mean(parents_scores, axis=0)
                metrics_std = np.std(parents_scores, axis=0)
                print("Standard deviation: ", metrics_std)

                #print(best_score,metrics_mean,metrics_std)
                # save best, mean and std scores
                with open(path.join(PATH_RESULTS, 'best_scores.csv'), 'a', newline='') as partial_file:
                    writer_csv = csv.writer(partial_file)
                    writer_csv.writerow(best_score_complete)
                with open(path.join(PATH_RESULTS, 'metrics_mean.csv'), 'a', newline='') as partial_file:
                    writer_csv = csv.writer(partial_file)
                    writer_csv.writerow(metrics_mean)
                with open(path.join(PATH_RESULTS, 'metrics_std.csv'), 'a', newline='') as partial_file:
                    writer_csv = csv.writer(partial_file)
                    writer_csv.writerow(metrics_std)
                for j in range(int(population_size/2)):
                    if j == 0 or select_parents_with_replacement:
                        pass
                    else:
                        popu_crossover.remove(parent1)
                        popu_crossover.remove(parent2)
                    # parents selection

                    parent1, parent2 = choose_parents_tournament(popu_crossover, None, fitness_eval_v02, num_per_tournament, True)
                    #print("parets_selection")
                    # cross-over
                    if Npoints:
                        child1, child2 = crossover_operator_Npoints(parent1, parent2, num_points, cross_prob)
                    else:
                        child1, child2 = crossover_operator_uniform(parent1, parent2, p_value, cross_prob)
                    #print("crossover")
                    # mutation
                    if parameter_mutation:
                        child1 = mutation_operator_by_parameter(child1, all_actions_by_type, mutation_prob)
                        child2 = mutation_operator_by_parameter(child2, all_actions_by_type, mutation_prob)
                    else:
                        child1 = mutation_operator_by_action(child1, all_actions, mutation_prob)
                        child2 = mutation_operator_by_action(child2, all_actions, mutation_prob)
                    #print("mutation")
                    offspring.append(child1)
                    offspring.append(child2)

                offspring_scores = np.array([fitness_eval_v02(individual, agent.request_game_reset(),False, 0) for individual in offspring])
                # survivor selection
                new_generation = steady_state_selection(population, parents_scores, offspring, offspring_scores, num_replace)
                population = new_generation
                generation += 1
                #print("survivor")
                print("\n")

        except Exception as e:
                print(f"Error: {e}")



        # calculate scores for last generation, and update files:

        last_generation_scores = np.array([fitness_eval_v02(individual,  agent.request_game_reset(),True, 0) for individual in population])
        index_best_score = np.argmax(last_generation_scores[:,0])
        best_score_complete = last_generation_scores[index_best_score, :]
        metrics_mean = np.mean(last_generation_scores, axis=0)
        metrics_std = np.std(last_generation_scores, axis=0)
        # save best, mean and std scores from last generation
        with open(path.join(PATH_RESULTS, 'best_scores.csv'), 'a', newline='') as partial_file:
            writer_csv = csv.writer(partial_file)
            writer_csv.writerow(best_score_complete)
        with open(path.join(PATH_RESULTS, 'metrics_mean.csv'), 'a', newline='') as partial_file:
            writer_csv = csv.writer(partial_file)
            writer_csv.writerow(metrics_mean)
        with open(path.join(PATH_RESULTS, 'metrics_std.csv'), 'a', newline='') as partial_file:
            writer_csv = csv.writer(partial_file)
            writer_csv.writerow(metrics_std)

        print("\nGeneration = ", generation)


        print("\nBest sequence score: ", best_score_complete)
       

        def save_population_json(agent, path):
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
            parsed_population_run = []
            for individual in agent.parsed_population:
                individual_str_list = []
                for action_result in individual:
                    if isinstance(action_result, tuple) and len(action_result) == 2:
                        # Format each action and result pair as a single string
                        action_str = f"[Action {str(action_result[0])}, {str(action_result[1])}]"
                        individual_str_list.append(action_str)
                    else:
                        # Handle cases where the structure isn't as expected
                        individual_str_list.append(str(action_result))

                parsed_population_run.append(individual_str_list)

            # Step 3: Append this run's population to the outer array
            outer_array.append(parsed_population_run)

            # Step 4: Write the updated outer array back to the file
            with open(path, "w") as f:
                json.dump(outer_array, f, indent=4)

        # Usage
        save_population_json(agent, os.path.join(PATH_RESULTS, 'parsed_population.json'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)

    args = parser.parse_args()
    
    agent = GeneticAgent(args.host, args.port,"Attacker")

    observation = agent.register()
    agent.play_game()
    agent._logger.info("Terminating interaction")
    agent.terminate_connection()
    