import json
import sys

def convert_workflow_to_jsonl(input_file, output_file):
    """
    Convert a JSON file containing workflows to JSONL format.
    
    Each workflow in the input JSON becomes a JSONL entry with messages for system, user, and assistant.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output JSONL file
    """
    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Open output file for writing
        with open(output_file, 'w', encoding='utf-8') as f:
            # Process each workflow entry
            for workflow_entry in data:
                # Extract workflow information
                workflow_json = workflow_entry.get("workflow", {})
                workflow_prompt = workflow_entry.get("prompt",{})
                
                # Create the JSONL entry structure
                jsonl_entry = {
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are an AI assistant that generates n8n workflow JSON configurations based on user prompts. Ensure the workflows are well-structured, executable, and aligned with the user's request."
                        },
                        {
                            "role": "user", 
                            "content": f"'{workflow_prompt}'"
                        },
                        {
                            "role": "assistant", 
                            "content": json.dumps(workflow_json)
                        }
                    ]
                }
                
                # Write the entry as a line in the JSONL file
                f.write(json.dumps(jsonl_entry) + '\n')
        
        print(f"Successfully converted {len(data)} workflows to JSONL format in {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Input file '{input_file}' contains invalid JSON.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_jsonl.py <input_json_file> <output_jsonl_file>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        convert_workflow_to_jsonl(input_file, output_file)