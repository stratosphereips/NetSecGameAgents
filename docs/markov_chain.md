
# Markov Chain Agent

This agent is designed to solve the NetSecGame using a Markov Chain approach. It selects the next action based on a set of transition probabilities and the previous action.

## Transition Probabilities
The transition probabilities are stored in `transition_probabilities.json`, which can be modified manually or generated from a dataset of solutions by a previous agent. Suggested transition probabilities were created using solutions from the Genetic Algorithm and the `solutions_to_matrix.py` utility.




## Utilities

- **filter_winning_solutions.py**: Filters only the winning solutions from a JSON file of solutions. Input and output paths can be set within the script or via arguments `-i` (input path) and `-o` (output path).

- **solutions_to_matrix.py**: Given a JSON file with solutions, it outputs a JSON file that represents the transition matrix. Input and output paths can be set within the script or provided as `-i` (input path) and `-o` (output path).

- **solutions_analyzer.py**: Analyzes a JSON file of solutions, displaying metrics such as average steps, win rate, and more. The input file can be set within the script or specified with `-d` (data path).

## Genetic Algorithm

A genetic algorithm is included within the Markov Chain Agent, which can be used to create the transition probabilities for the Markov Chain Agent.

This algorithm requires that the starting position and goal be fixed in the environment configuration to evolve between generations.

All parameters for the Genetic Algorithm can be adjusted in the `config.json` file.

### Parameter List:


- **Population size**: the number of individuals in the population (tested at 2500).


- **Number of Generations**:
    - Maximum number of generations (tested at 55).
    - Fitness function threshold (tested at 17500).

- **Mutation**:
    - Mutation probability: chance of a given action mutating (tested at 0.0333).
    - Type of mutation: can be mutation by parameters or mutation by action (tested with mutation by action).

- **Crossover**:
    - Crossover probability (tested at 1).
    - N-points crossover (tested at 6).

- **Replacement**: Enables an elitist approach. The number of individuals that remain unchanged between generations can be specified (tested at 50).
