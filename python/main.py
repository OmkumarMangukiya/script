import json
import argparse
import os
import requests
import re
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
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def prepare_chat_messages(self, workflow: Dict[str, Any]) -> List[Dict[str, str]]:
        """Prepares messages for OpenAI API."""
        workflow_name = workflow.get("name", "Unnamed Workflow")
        simplified_workflow = {
            "name": workflow_name,
            "nodes": [{"id": node["id"], "name": node["name"], "type": node["type"]} for node in workflow.get("nodes", [])],
            "connections": workflow.get("connections", {})
        }
        system_message = "You are an expert in N8N workflows. Generate training prompts for the given workflow that represent how users might request this automation."

        user_message = f"""
Workflow Name: {workflow_name}

Workflow JSON:
{json.dumps(simplified_workflow, indent=2)}

Generate exactly 3 different training prompts that would elicit this workflow as a response:
- 2 prompts should be direct commands/instructions (e.g., "Create an N8N workflow that...", "Build an automation to...")
- 1 prompt should be a natural question that a user might actually ask (e.g., "How can I automate...", "Is it possible to connect...")

Make the prompts diverse and realistic - think about what a real user might ask or request when they need this workflow.

IMPORTANT: Return only a plain JSON array without any markdown formatting or code blocks.
The response should look exactly like this:
["Command Prompt 1", "Command Prompt 2", "Question Prompt"]

Do not include any explanations, code blocks (like ```json), or extra text.
        """

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    def extract_json_from_response(self, content: str) -> str:
        """Extracts JSON content from a response that might contain markdown code blocks."""
        # Check if content is wrapped in markdown code blocks
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(code_block_pattern, content)
        
        if match:
            # Extract content from code block
            return match.group(1).strip()
        else:
            # Return original content if no code block is found
            return content.strip()
    
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
            "temperature": 0.9,
            "max_tokens": 600,
            "top_p": 0.95
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()

            # Debugging: Print full response
            print("OpenAI Response:", json.dumps(response_json, indent=2))

            if "choices" not in response_json or not response_json["choices"]:
                raise ValueError("OpenAI API response does not contain expected 'choices' field.")

            content = response_json["choices"][0]["message"]["content"].strip()
            
            # Extract JSON content from potential markdown code blocks
            extracted_content = self.extract_json_from_response(content)
            
            try:
                prompts = json.loads(extracted_content)
                if not isinstance(prompts, list) or len(prompts) != 3:
                    raise ValueError(f"OpenAI response is not a valid list of 3 prompts. Response: {extracted_content}")
                return prompts
            except json.JSONDecodeError:
                raise ValueError(f"OpenAI returned invalid JSON after extraction. Original response: {content}\nExtracted content: {extracted_content}")
        
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