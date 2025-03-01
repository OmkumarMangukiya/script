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
        Prepare ChatGPT API messages to generate training prompts from a workflow.
        
        Args:
            workflow: N8N workflow dictionary
            
        Returns:
            List of message dictionaries for ChatGPT API
        """
        # Extract workflow information
        workflow_name = workflow.get("name", "Unnamed Workflow")
        
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
            "and generate a set of 5 different training prompts. These prompts should be phrased as "
            "direct requests that would elicit this workflow as a response. For example, if the workflow "
            "is 'chat with postgresql', prompts should be like 'give me an AI agent which helps me to chat "
            "with postgresql' or 'workflow to talk with postgresql with these functionalities...'."
        )
        
        # Prepare the user message with workflow details
        user_message = f"""
Workflow Name: {workflow_name}

Workflow JSON:
{json.dumps(simplified_workflow, indent=2)}

Please generate 5 different training prompts that are phrased as direct requests for this workflow.
Each prompt should:
- Start with phrases like "Create a workflow that...", "I need an AI agent that...", "Build me a system to..."
- Directly reference the main functionality in the workflow name
- Be specific about what capabilities are needed
- Vary in complexity and focus
- Be formatted as plain text without numbering or special formatting

Example for a workflow named "Chat with PostgreSQL Database":
"Give me an AI agent which helps me to chat with PostgreSQL database"
"Create a workflow that lets non-technical users query PostgreSQL using natural language"
"Build me a system to interact with my PostgreSQL database through conversation"
"""
        
        # Construct the messages array for the API call
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        return messages
    
    def generate_prompts(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate training prompts using OpenAI API.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Generated prompts
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
    
    def parse_response(self, response: str) -> List[str]:
        """
        Parse the response to extract prompts.
        
        Args:
            response: Response text from OpenAI API
            
        Returns:
            List of prompts
        """
        # Split by newlines and filter out empty lines and lines with just numbers
        lines = response.split("\n")
        prompts = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Skip lines that are just numbers or bullets
            if line.startswith(("1.", "2.", "3.", "4.", "5.", "-", "*", "•")):
                line = line[line.find(" ") + 1:].strip()
            # Skip lines that start with "Prompt" 
            if line.lower().startswith(("prompt")):
                colon_pos = line.find(":")
                if colon_pos != -1:
                    line = line[colon_pos + 1:].trip()
            # Skip quote marks
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1].strip()
            
            # Add to prompts if not empty after processing
            if line and not line.lower().startswith(("here are", "training prompts", "examples")):
                prompts.append(line)
        
        return prompts
    
    def create_jsonl_entry(self, workflow: Dict[str, Any], prompts: List[str]) -> Dict[str, Any]:
        """
        Create a JSONL entry for fine-tuning with prompts and workflow JSON.
        
        Args:
            workflow: Workflow dictionary
            prompts: List of training prompts
            
        Returns:
            JSONL entry dictionary
        """
        # Create a clean copy of the workflow
        clean_workflow = copy.deepcopy(workflow)
        
        # Create the JSONL entry with prompts and workflow JSON
        return {
            "workflow_name": workflow.get("name", "Unnamed Workflow"),
            "training_prompts": prompts,
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
        
        # Open output file for appending
        with open(output_file, 'a') as f:
            # Process each workflow
            for i, workflow in enumerate(workflows):
                print(f"Processing workflow {i+1}/{len(workflows)}: {workflow.get('name', 'Unnamed')}")
                
                # Prepare messages for ChatGPT
                messages = self.prepare_chat_messages(workflow)
                
                # Generate prompts
                response = self.generate_prompts(messages)
                
                # Parse the response
                prompts = self.parse_response(response)
                
                # Create JSONL entry
                jsonl_entry = self.create_jsonl_entry(workflow, prompts)
                
                # Write to output file
                f.write(json.dumps(jsonl_entry) + '\n')
                
                print(f"Processed workflow: {workflow.get('name', 'Unnamed')}")
                print(f"Generated {len(prompts)} training prompts:")
                for idx, prompt in enumerate(prompts[:3]):
                    print(f"  {idx+1}. {prompt[:80]}...")
                if len(prompts) > 3:
                    print(f"  ... and {len(prompts) - 3} more")
        
        print(f"Successfully processed {len(workflows)} workflows. Output written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Process N8N workflows to generate training prompts for fine-tuning')
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