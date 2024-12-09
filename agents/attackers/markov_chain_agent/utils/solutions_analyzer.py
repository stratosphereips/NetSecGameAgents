import json
import statistics
import argparse

# Default file path for data
DEFAULT_DATA_PATH = ''

def calculate_statistics(data, shortest_only=False):
    """Calculate average, standard deviation, and absolute shortest action counts."""
    action_counts = []
    absolute_shortest = float('inf')  # Initialize with infinity

    for outer_list in data:
        if shortest_only:
            # Find shortest sequence in current outer list
            shortest_sequence = min(outer_list, key=len)
            sequence_length = len(shortest_sequence)
            action_counts.append(sequence_length)
            absolute_shortest = min(absolute_shortest, sequence_length)
        else:
            # Process all sequences
            for inner_list in outer_list:
                sequence_length = len(inner_list)
                action_counts.append(sequence_length)
                absolute_shortest = min(absolute_shortest, sequence_length)

    # Calculate average and standard deviation
    average_actions = statistics.mean(action_counts)
    std_dev_actions = statistics.stdev(action_counts) if len(action_counts) > 1 else 0

    return average_actions, std_dev_actions, absolute_shortest

def has_final_nine(inner_list):
    """Check if the last action in the inner list contains ', 9]'."""
    if inner_list:
        last_action = inner_list[-1]
        return ', 9]' in last_action
    return False

def calculate_winning_percentage(data, shortest_only):
    """Calculate winning percentage based on the shortest_only flag."""
    total_individuals = 0
    winning_individuals = 0

    for outer_list in data:
        if shortest_only:
            # Only check the shortest sequence in the current outer list
            shortest_sequence = min(outer_list, key=len)
            total_individuals += 1
            if has_final_nine(shortest_sequence):
                winning_individuals += 1
        else:
            # Check all sequences in the current outer list
            for inner_list in outer_list:
                total_individuals += 1
                if has_final_nine(inner_list):
                    winning_individuals += 1

    return (winning_individuals / total_individuals) * 100 if total_individuals > 0 else 0.0

def main(data_path):
    """Main function to load data and calculate statistics."""
    # Load the JSON data from the file
    with open(data_path, 'r') as file:
        data = json.load(file)

    # Calculate and print statistics for both shortest_only = True and False
    for shortest_only in [True, False]:
        average, std_dev, absolute_shortest = calculate_statistics(data, shortest_only)
        average, std_dev = round(average, 2), round(std_dev, 2)
        winning_percentage = calculate_winning_percentage(data, shortest_only)

        if shortest_only:
            print("Results for Shortest Only:")
            print(f'Average number of actions in shortest sequences: {average}')
            print(f'Standard deviation of actions in shortest sequences: {std_dev}')
        else:
            print("Results for All Sequences:")
            print(f'Average number of actions per sequence: {average}')
            print(f'Standard deviation of actions per sequence: {std_dev}')

        print(f'Absolute shortest sequence length: {absolute_shortest}')
        print(f"Winning Percentage: {winning_percentage:.2f}%\n")

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process JSON data path.')
    parser.add_argument('-d', '--data', default=DEFAULT_DATA_PATH, help='Data file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function with the provided or default path
    main(args.data)
