import json
import openai
import os

# Set your OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

def generate_prompt(name, description):
    """Uses OpenAI API to generate a natural-sounding prompt based on name and description. Think like a user describing what they need—make it intuitive and natural"""
    system_message = "You generate user prompts for an AI that builds n8n workflows from natural language."
    
    user_message = (
        f"Given the following workflow details, generate a natural-sounding prompt that a user might give to an AI workflow generator.\n\n"
        f"Workflow Name: {name}\n"
        f"Description: {description}\n\n"
        f"The prompt should be concise (under 50 words), and simple too"
        f"also dont user something like this  Create a workflow named` Extract and process information directly from PDF using Claude and Gemini` like dont give name"
        f"Use direct commands like 'Create a' or 'Build a' instead of questions"
    )
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_message},
                  {"role": "user", "content": user_message}],
        max_tokens=500,
        temperature=0.9
    )
    
    return response.choices[0].message.content.strip()

def main():
    input_filename = "input.json"
    output_filename = "output.jsonl"
    
    # Load workflow data
    with open(input_filename, "r", encoding="utf-8") as infile:
        workflows = json.load(infile)
    
    with open(output_filename, "a", encoding="utf-8") as outfile:
        for wf in workflows:
            name = wf.get("name", "Unnamed Workflow")
            description = wf.get("description", "")
            workflow_json = wf.get("workflow", {})
            
            # Generate user prompt using OpenAI API
            user_prompt = generate_prompt(name, description)
            
            # Remove surrounding quotes if present
            if user_prompt.startswith('"') and user_prompt.endswith('"'):
                user_prompt = user_prompt[1:-1]
            
            # Prepare JSONL entry
            training_entry = {
                "messages": [
                    {"role": "system", "content": "You are an AI-powered n8n workflow builder. You generate n8n workflows in JSON format based on user prompts."},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": json.dumps(workflow_json)}
                ]
            }
            
            # Write to JSONL file
            outfile.write(json.dumps(training_entry) + "\n")

if __name__ == "__main__":
    main()
