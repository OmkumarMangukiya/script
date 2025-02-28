import json
import argparse
import os
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv
import copy

# Load environment variables from .env file
load_dotenv()

class N8NWorkflowProcessor:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the workflow processor with OpenAI API key.
        
        Args:
            api_key: OpenAI API key (optional, can be set as environment variable)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it as an argument or OPENAI_API_KEY environment variable.")
    
    def load_workflows(self, input_file: str) -> List[Dict[str, Any]]:
        """
        Load N8N workflows from a JSON file.
        
        Args:
            input_file: Path to the input JSON file
            
        Returns:
            List of workflow dictionaries
        """
        try:
            with open(input_file, 'r') as f:
                workflows = json.load(f)
            return workflows
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {input_file}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file {input_file} not found")
    
    def prepare_chat_messages(self, workflow: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Prepare ChatGPT API messages from a workflow.
        
        Args:
            workflow: N8N workflow dictionary
            
        Returns:
            List of message dictionaries for ChatGPT API
        """
        # Extract workflow information
        workflow_name = workflow.get("name", "Unnamed Workflow")
        workflow_description = workflow.get("description", "No description provided")
        workflow_others = workflow.get("others", "")
        
        # Create a simplified version of the workflow that includes only essential elements
        simplified_workflow = {
            "name": workflow_name,
            "nodes": [
                {
                    "id": node["id"],
                    "name": node["name"],
                    "type": node["type"],
                    "position": node["position"],
                    "parameters": node.get("parameters", {})
                }
                for node in workflow.get("nodes", [])
            ],
            "connections": workflow.get("connections", {})
        }
        
        # Prepare the system message
        system_message = (
            "You are an expert on N8N workflows. Your task is to analyze the provided workflow "
            "and generate a comprehensive, step-by-step explanation of how it works. "
            "Focus on the purpose, functionality, and dataflow between nodes."
        )
        
        # Prepare the user message with workflow details
        user_message = f"""
Workflow Name: {workflow_name}

Description: {workflow_description}

Additional Information: {workflow_others}

Workflow JSON:
{json.dumps(simplified_workflow, indent=2)}

Please analyze this N8N workflow and provide:
1. A clear overview of what this workflow accomplishes
2. A detailed explanation of how data flows between nodes
3. Key configuration details and parameters
4. Any prerequisites or setup requirements
"""
        
        # Construct the messages array for the API call
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        return messages
    
    def generate_explanation(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate an explanation using OpenAI API.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Generated explanation text
        """
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {str(e)}")
    
    def create_jsonl_entry(self, workflow: Dict[str, Any], explanation: str) -> Dict[str, Any]:
        """
        Create a JSONL entry for fine-tuning with a simplified prompt and workflow JSON.
        
        Args:
            workflow: Workflow dictionary
            explanation: Generated explanation
            
        Returns:
            JSONL entry dictionary with prompt and workflow JSON
        """
        # Create a clean copy of the workflow without description and others fields
        clean_workflow = copy.deepcopy(workflow)
        if "description" in clean_workflow:
            del clean_workflow["description"]
        if "others" in clean_workflow:
            del clean_workflow["others"]
        
        # Create a simple, concise prompt
        workflow_name = workflow.get("name", "Unnamed Workflow")
        prompt = f"Explain how this n8n workflow works: {workflow_name}"
        
        # Create the JSONL entry with prompt, completion, and workflow JSON
        return {
            "prompt": prompt,
            "completion": explanation,
            "workflow": clean_workflow
        }
    
    def process_workflows(self, input_file: str, output_file: str):
        """
        Process workflows from input file and write JSONL to output file.
        
        Args:
            input_file: Path to the input JSON file
            output_file: Path to the output JSONL file
        """
        # Load workflows
        workflows = self.load_workflows(input_file)
        
        # Open output file for writing
        with open(output_file, 'w') as f:
            # Process each workflow
            for i, workflow in enumerate(workflows):
                print(f"Processing workflow {i+1}/{len(workflows)}: {workflow.get('name', 'Unnamed')}")
                
                # Prepare messages for ChatGPT
                messages = self.prepare_chat_messages(workflow)
                
                # Generate explanation
                explanation = self.generate_explanation(messages)
                
                # Create JSONL entry
                jsonl_entry = self.create_jsonl_entry(workflow, explanation)
                
                # Write to output file
                f.write(json.dumps(jsonl_entry) + '\n')
                
                print(f"Processed workflow: {workflow.get('name', 'Unnamed')}")
        
        print(f"Successfully processed {len(workflows)} workflows. Output written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Process N8N workflows to generate JSONL files for fine-tuning')
    parser.add_argument('--input', '-i', required=True, help='Input JSON file containing N8N workflows')
    parser.add_argument('--output', '-o', required=True, help='Output JSONL file for fine-tuning')
    parser.add_argument('--api-key', '-k', help='OpenAI API key (alternatively, set OPENAI_API_KEY environment variable)')
    
    args = parser.parse_args()
    
    try:
        processor = N8NWorkflowProcessor(api_key=args.api_key)
        processor.process_workflows(args.input, args.output)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())