import json
import sys

def convert_jsonl_to_json(input_file, output_file):
    """
    Convert a JSONL file back to a JSON array format.
    
    Args:
        input_file (str): Path to the input JSONL file
        output_file (str): Path to the output JSON file
    """
    try:
        # Initialize an empty array to hold the converted data
        workflows = []
        
        # Read the JSONL file line by line
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    # Parse the JSONL entry
                    entry = json.loads(line)
                    
                    # Extract the necessary information
                    system_content = entry.get("messages", [])[0].get("content", "") if len(entry.get("messages", [])) > 0 else ""
                    user_content = entry.get("messages", [])[1].get("content", "") if len(entry.get("messages", [])) > 1 else ""
                    assistant_content = entry.get("messages", [])[2].get("content", "") if len(entry.get("messages", [])) > 2 else ""
                    
                    
                    # Try to parse the workflow JSON from the assistant content
                    try:
                        workflow_json = json.loads(assistant_content)
                    except json.JSONDecodeError:
                        # If the assistant content isn't valid JSON, use it as is
                        workflow_json = assistant_content
                    
                    # Create the workflow entry
                    workflow_entry = {
            
                        "workflow": workflow_json
                    }
                    
                    # Add to the workflows array
                    workflows.append(workflow_entry)
        
        # Write the JSON array to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(workflows, f, indent=2)
        
        print(f"Successfully converted JSONL to JSON. Found {len(workflows)} workflows.")
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in the input file: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_jsonl_to_json.py <input_jsonl_file> <output_json_file>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        convert_jsonl_to_json(input_file, output_file)