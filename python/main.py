import json
import sys
import requests

def call_openai_api(prompt, api_key, model="gpt-4o", temperature=0.9, max_tokens=500):
    """
    Call OpenAI API directly using the requests library.
    
    Args:
        prompt (list): List of message dictionaries with role and content
        api_key (str): OpenAI API key
        model (str): The model to use, default is gpt-4o
        temperature (float): Controls randomness, default is 0.9
        max_tokens (int): Maximum tokens in response, default is 500
        
    Returns:
        str: The content of the generated response
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model,
        "messages": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Ensure HTTP errors are caught
        response_json = response.json()
        
        if "choices" not in response_json or not response_json["choices"]:
            raise ValueError("OpenAI API response does not contain expected 'choices' field.")
        
        content = response_json["choices"][0]["message"]["content"].strip()
        return content
    
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to connect to OpenAI API: {str(e)}")

def generate_high_quality_prompt(workflow_name, workflow_description, api_key):
    """
    Generate a high-quality user prompt for the workflow using the OpenAI API.
    
    Args:
        workflow_name (str): Name of the workflow
        workflow_description (str): Description of the workflow
        api_key (str): OpenAI API key
        
    Returns:
        str: Enhanced user prompt
    """
    # Create a more detailed system prompt with guidelines for generating good prompts
    system_prompt = """
    You are an expert in n8n automation workflows. Your task is to generate high-quality, natural-sounding prompts
    that would lead someone to create the workflow described. The prompts should:
    
    1. Be specific about what the workflow should accomplish
    2. Mention key integrations or services that should be used
    3. Include any important workflow logic or conditional steps
    4. Sound natural, as if written by a real user with a business need
    5. Be concise yet detailed (40-70 words), without JSON or technical jargon
    6. "Use direct commands like 'Create a' or 'Build a' instead of questions.
    
    
    Think like a user explaining their workflow request—make it sound natural and intuitive!
    """
    
    user_content = f"""
    Create a natural-sounding prompt based on this workflow:
    
    Name: {workflow_name}
    
    Description: {workflow_description}
    
    The prompt should describe what a user would request to have this workflow built.
    """
    
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    try:
        return call_openai_api(prompt_messages, api_key)
    except ValueError as e:
        print(f"Warning: Could not generate prompt for '{workflow_name}': {str(e)}")
        # Return a basic fallback prompt using the available information
        return f"Create an n8n workflow for {workflow_name} that {workflow_description}"

def convert_workflow_to_jsonl(input_file, output_file, api_key=None):
    """
    Convert a JSON file containing workflows to JSONL format.
    
    Each workflow in the input JSON becomes a JSONL entry with messages for system, user, and assistant.
    If api_key is provided, it will use the OpenAI API to generate better user prompts.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output JSONL file
        api_key (str, optional): OpenAI API key for enhancing workflows
    """
    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Open output file for writing
        with open(output_file, 'w', encoding='utf-8') as f:
            # Process each workflow entry
            for i, workflow_entry in enumerate(data):
                # Extract workflow information
                workflow_name = workflow_entry.get("name", "Unnamed Workflow")
                workflow_description = workflow_entry.get("description", "No description")
                workflow_json = workflow_entry.get("workflow", {})
                
                # Generate user prompt
                if api_key:
                    try:
                        user_prompt = generate_high_quality_prompt(workflow_name, workflow_description, api_key)
                        print(f"[{i+1}/{len(data)}] Generated enhanced prompt for: {workflow_name}")
                    except Exception as e:
                        print(f"[{i+1}/{len(data)}] Error generating prompt for {workflow_name}: {str(e)}")
                        user_prompt = f"Create an n8n workflow called '{workflow_name}' that {workflow_description}"
                else:
                    # Basic prompt if no API key provided
                    user_prompt = f"Create an n8n workflow called '{workflow_name}' that {workflow_description}"
                
                # Create the JSONL entry structure
                jsonl_entry = {
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are an AI assistant that generates n8n workflow JSON configurations based on user prompts. Ensure the workflows are well-structured, executable, and aligned with the user's request."
                        },
                        {
                            "role": "user", 
                            "content": user_prompt
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
    if len(sys.argv) < 3:
        print("Usage: python convert_to_jsonl.py <input_json_file> <output_jsonl_file> [openai_api_key]")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        api_key = sys.argv[3] if len(sys.argv) > 3 else None
        convert_workflow_to_jsonl(input_file, output_file, api_key)