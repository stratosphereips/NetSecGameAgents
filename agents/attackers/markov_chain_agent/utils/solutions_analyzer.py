import json
import statistics
import argparse

# Default file path for data
DEFAULT_DATA_PATH = ''

def calculate_statistics(data, shortest_only=False):
    """Calculate average, standard deviation, and absolute shortest action counts."""
    action_counts = []
    absolute_shortest = float('inf')
    valid_shortest_sequences = []
    
    for outer_list in data:
        # Filter sequences with final 9
        valid_sequences = [
            seq for seq in outer_list 
            if has_final_nine(seq)
        ]
        
        if shortest_only:
            # Only keep the shortest valid sequence per outer_list
            if valid_sequences:
                shortest_sequence = min(valid_sequences, key=len)
                action_counts.append(len(shortest_sequence))
                valid_shortest_sequences.append(shortest_sequence)
        else:
            # Keep lengths of all valid sequences
            for seq in valid_sequences:
                action_counts.append(len(seq))
        
        # Update absolute shortest across all valid sequences
        if valid_sequences:
            absolute_shortest = min(absolute_shortest, min(len(seq) for seq in valid_sequences))
    
    # Calculate statistics
    average_actions = statistics.mean(action_counts) if action_counts else 0
    std_dev_actions = statistics.stdev(action_counts) if len(action_counts) > 1 else 0
    
    return average_actions, std_dev_actions, absolute_shortest, valid_shortest_sequences

def has_final_nine(inner_list):
    """Check if the last action in the inner list contains ', 9]'."""
    if inner_list:
        last_action = inner_list[-1]
        return ', 9]' in last_action
    return False

def has_detection(inner_list):
    """Check if the last action in the inner list contains ', -9]'."""
    if inner_list:
        last_action = inner_list[-1]
        return ', -9]' in last_action
    return False

def calculate_winning_percentage(data, shortest_only):
    """Calculate winning percentage and detection rate based on the shortest_only flag."""
    total_individuals = 0
    winning_individuals = 0
    detection_individuals = 0
    
    for outer_list in data:
        if shortest_only:
            # Only check the shortest sequence in the current outer list
            valid_sequences = [
                seq for seq in outer_list 
                if has_final_nine(seq)
            ]
            
            if valid_sequences:
                total_individuals += 1
                shortest_sequence = min(valid_sequences, key=len)
                
                if has_final_nine(shortest_sequence):
                    winning_individuals += 1
        
        else:
            # Check all sequences in the current outer list
            for inner_list in outer_list:
                total_individuals += 1
                
                if has_final_nine(inner_list):
                    winning_individuals += 1
                
                if has_detection(inner_list):
                    detection_individuals += 1
    
    winning_percentage = (winning_individuals / total_individuals) * 100 if total_individuals > 0 else 0.0
    detection_percentage = (detection_individuals / total_individuals) * 100 if total_individuals > 0 else 0.0
    
    return winning_percentage, detection_percentage, total_individuals

def main(data_path):
    """Main function to load data and calculate statistics."""
    # Load the JSON data from the file
    with open(data_path, 'r') as file:
        data = json.load(file)
    
    # Calculate and print statistics for both shortest_only = True and False
    for shortest_only in [True, False]:
        average, std_dev, absolute_shortest, valid_shortest_sequences = calculate_statistics(data, shortest_only)
        average, std_dev = round(average, 2), round(std_dev, 2)
        
        winning_percentage, detection_percentage, total_individuals = calculate_winning_percentage(data, shortest_only)
        
        if shortest_only:
            print("Results for Shortest Only:")
            print(f'Average number of actions in valid shortest sequences: {average}')
            print(f'Standard deviation of actions in valid shortest sequences: {std_dev}')
            print(f'Number of valid shortest sequences: {len(valid_shortest_sequences)}')
        else:
            print("Results for All Sequences:")
            print(f'Average number of actions per sequence: {average}')
            print(f'Standard deviation of actions per sequence: {std_dev}')
        
        print(f'Absolute shortest sequence length: {absolute_shortest}')
        print(f"Total Individuals Analyzed: {total_individuals}")
        print(f"Winning Percentage: {winning_percentage:.2f}%")
        print(f"Detection Percentage: {detection_percentage:.2f}%\n")

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process JSON data path.')
    parser.add_argument('-d', '--data', default=DEFAULT_DATA_PATH, help='Data file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function with the provided or default path
    main(args.data)