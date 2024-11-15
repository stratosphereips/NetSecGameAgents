import json
import argparse

# Default file paths
DEFAULT_INPUT_PATH = ''
DEFAULT_OUTPUT_PATH = ''

def has_final_nine(inner_list):
    """Check if the last action in the inner list contains ', 9]'."""
    if inner_list:
        last_action = inner_list[-1]
        return ', 9]' in last_action
    return False

def process_inner_list(inner_list):
    """Process each inner list of actions, stopping if ', 9]' is encountered."""
    result = []
    for action in inner_list:
        result.append(action)
        # Stop copying if we find the action with ', 9]'
        if ', 9]' in action:
            break
    return result

def process_json(data):
    """Process the entire JSON structure to extract relevant actions."""
    processed_data = []
    for outer_list in data:
        processed_outer_list = [
            process_inner_list(inner_list) 
            for inner_list in outer_list if has_final_nine(inner_list)
        ]
        if processed_outer_list:
            processed_data.append(processed_outer_list)
    return processed_data

def main(input_path, output_path):
    """Main function to read, process, and save JSON data."""
    # Read the JSON file
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Process the data
    processed_data = process_json(data)

    # Write the processed data to a new JSON file
    with open(output_path, 'w') as file:
        json.dump(processed_data, file, indent=4)

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process JSON input and output paths.')
    parser.add_argument('-i', '--input', default=DEFAULT_INPUT_PATH, help='Input file path')
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT_PATH, help='Output file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function with provided or default paths
    main(args.input, args.output)
