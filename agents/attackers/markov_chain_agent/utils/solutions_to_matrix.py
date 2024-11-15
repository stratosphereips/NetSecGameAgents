import json
from collections import defaultdict, Counter
import argparse

# Default file paths
DEFAULT_INPUT_FILE_PATH = ''
DEFAULT_OUTPUT_FILE_PATH = ''

def extract_action_types(sequence):
    """Extract action types from a sequence of actions."""
    return [action.split('<')[1].split('|')[0].split('.')[1] for action in sequence]

def calculate_transitions(sequences):
    """Calculate transitions, initial, and final counts from action sequences."""
    action_types = set()
    transitions = defaultdict(Counter)
    initial_counts = Counter()
    final_counts = Counter()
    total_actions_handled = 0

    for sequence_set in sequences:
        for sequence in sequence_set:
            action_list = extract_action_types(sequence)
            initial_counts[action_list[0]] += 1
            final_counts[action_list[-1]] += 1
            
            for i in range(len(action_list) - 1):
                from_type = action_list[i]
                to_type = action_list[i + 1]
                transitions[from_type][to_type] += 1
                total_actions_handled += 1
                
            action_types.update(action_list)

    return transitions, initial_counts, final_counts, total_actions_handled, action_types

def calculate_probabilities(transitions, initial_counts, final_counts, total_initial, total_final, column_order):
    """Compute transition, initial, and final probabilities."""
    matrix = defaultdict(lambda: defaultdict(float))
    
    for from_type in column_order[1:-1]:  # Exclude "Initial Action" and "Final Probability"
        total_from_type = sum(transitions[from_type].values())
        for to_type in column_order[1:-1]:
            count = transitions[from_type].get(to_type, 0)
            matrix[from_type][to_type] = round(count / total_from_type, 2) if total_from_type > 0 else 0

    initial_prob = {action_type: round(count / total_initial, 2) for action_type, count in initial_counts.items()}
    final_prob = {action_type: round(count / total_final, 2) for action_type, count in final_counts.items()}

    return matrix, initial_prob, final_prob

def build_json_data(matrix, initial_prob, final_prob, column_order):
    """Prepare the JSON data structure for transition probabilities."""
    json_data = []

    for action_type in column_order[1:-1]:  # Exclude "Initial Action" and "Final Probability"
        action_entry = {
            "Action": action_type,
            "ScanNetwork": round(matrix[action_type].get("ScanNetwork", 0), 2),
            "FindServices": round(matrix[action_type].get("FindServices", 0), 2),
            "ExploitService": round(matrix[action_type].get("ExploitService", 0), 2),
            "FindData": round(matrix[action_type].get("FindData", 0), 2),
            "ExfiltrateData": round(matrix[action_type].get("ExfiltrateData", 0), 2),
            "FinalProbability": round(final_prob.get(action_type, 0), 2)
        }
        json_data.append(action_entry)

    initial_entry = {
        "Action": "Initial Action",
        "ScanNetwork": round(initial_prob.get("ScanNetwork", 0), 2),
        "FindServices": round(initial_prob.get("FindServices", 0), 2),
        "ExploitService": round(initial_prob.get("ExploitService", 0), 2),
        "FindData": round(initial_prob.get("FindData", 0), 2),
        "ExfiltrateData": round(initial_prob.get("ExfiltrateData", 0), 2),
        "FinalProbability": 0  
    }
    json_data.insert(0, initial_entry)
    
    return json_data

def main(input_file_path, output_file_path):
    """Main function to process JSON data and calculate transition probabilities."""
    # Load JSON data from file
    with open(input_file_path, 'r') as file:
        sequences = json.load(file)
    
    # Initialize counters and process transitions
    column_order = [
        "Initial Action",
        "ScanNetwork",
        "FindServices",
        "ExploitService",
        "FindData",
        "ExfiltrateData",
        "Final Probability"
    ]
    
    transitions, initial_counts, final_counts, total_actions_handled, action_types = calculate_transitions(sequences)
    
    total_initial = sum(initial_counts.values())
    total_final = sum(final_counts.values())
    
    # Compute transition probabilities
    matrix, initial_prob, final_prob = calculate_probabilities(
        transitions, initial_counts, final_counts, total_initial, total_final, column_order
    )

    # Build JSON data structure
    json_data = build_json_data(matrix, initial_prob, final_prob, column_order)

    # Write to output JSON file
    with open(output_file_path, 'w') as output_file:
        json.dump({"transition_probabilities": json_data}, output_file, indent=4)
    
    # Print total actions handled
    #print(f"\nTotal actions handled: {total_actions_handled}")

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process input and output JSON file paths.')
    parser.add_argument('-i', '--input', default=DEFAULT_INPUT_FILE_PATH, help='Input JSON file path')
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT_FILE_PATH, help='Output JSON file path')

    # Parse arguments
    args = parser.parse_args()

    # Run the main function with provided or default paths
    main(args.input, args.output)
