import json
import argparse
import os
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Load environment variables from .env file
load_dotenv()

class N8NWorkflowProcessor:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it as an argument or OPENAI_API_KEY environment variable.")
    
    def load_workflows(self, input_file: str) -> List[Dict[str, Any]]:
        """Loads workflows from a JSON file."""
        with open(input_file, 'r') as f:
            return json.load(f)
    
    def prepare_chat_messages(self, workflow: Dict[str, Any]) -> List[Dict[str, str]]:
        """Prepares messages for OpenAI API."""
        workflow_name = workflow.get("name", "Unnamed Workflow")
        simplified_workflow = {
            "name": workflow_name,
            "nodes": [{"id": node["id"], "name": node["name"], "type": node["type"]} for node in workflow.get("nodes", [])],
            "connections": workflow.get("connections", {})
        }
        system_message = "You are an expert in N8N workflows. Generate exactly 4 diverse training prompts for the given workflow."

        user_message = f"""
Workflow Name: {workflow_name}

Workflow JSON:
{json.dumps(simplified_workflow, indent=2)}

Generate exactly 4 different training prompts that would elicit this workflow as a response.
Return only a JSON array with the structure:
["Prompt 1", "Prompt 2", "Prompt 3" , "Prompt 4"]
Do not include any explanations or extra text.
        """

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    def generate_prompts(self, messages: List[Dict[str, str]]) -> List[str]:
        """Calls OpenAI API to generate structured prompts with error handling."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": "gpt-4o",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Ensure HTTP errors are caught
            response_json = response.json()

            # Debugging: Print full response
            print("OpenAI Response:", json.dumps(response_json, indent=2))

            if "choices" not in response_json or not response_json["choices"]:
                raise ValueError("OpenAI API response does not contain expected 'choices' field.")

            content = response_json["choices"][0]["message"]["content"].strip()

            try:
                prompts = json.loads(content)
                if not isinstance(prompts, list) or len(prompts) != 4:
                    raise ValueError(f"OpenAI response is not a valid list of 4 prompts. Response: {content}")
                return prompts
            except json.JSONDecodeError:
                raise ValueError(f"OpenAI returned invalid JSON. Response: {content}")
        
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to connect to OpenAI API: {str(e)}")

    def create_jsonl_entries(self, workflow: Dict[str, Any], prompts: List[str]) -> List[Dict[str, Any]]:
        """Creates JSONL formatted entries for fine-tuning."""
        return [{
            "messages": [
                {"role": "system", "content": "You are an AI assistant helping users create N8N workflows."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(workflow)}
            ]
        } for prompt in prompts]
    
    def process_workflows(self, input_file: str, output_file: str):
        """Processes workflows and saves results to JSONL file."""
        workflows = self.load_workflows(input_file)
        
        with open(output_file, 'a') as f:
            for workflow in workflows:
                messages = self.prepare_chat_messages(workflow)
                prompts = self.generate_prompts(messages)
                jsonl_entries = self.create_jsonl_entries(workflow, prompts)
                
                for entry in jsonl_entries:
                    f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process N8N workflows into fine-tuning JSONL format")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file containing N8N workflows")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file for fine-tuning")
    parser.add_argument("--api-key", "-k", help="OpenAI API key (or set OPENAI_API_KEY env variable)")
    
    args = parser.parse_args()
    processor = N8NWorkflowProcessor(api_key=args.api_key)
    processor.process_workflows(args.input, args.output)
