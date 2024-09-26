import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")

def generate_response(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**input_ids, max_new_tokens=256)
    response = tokenizer.decode(outputs[0])
    print(outputs)
    return response

      
def generate_prompt():
    prompt = """User: Ты — старший разработчик, и даешь задание своему помощнику, который только учится программировать. 
Помощник умеет выполнять следующие команды:
1. Google Search: "google", args: "input": "<search>"
2. Browse Website: "browse_website", args: "url": "<url>", "question": "<what_you_want_to_find_on_website>"
3. Start GPT Agent: "start_agent", args: "name": "<name>", "task": "<short_task_desc>", "prompt": "<prompt>"
4. Message GPT Agent: "message_agent", args: "key": "<key>", "message": "<message>"
5. List GPT Agents: "list_agents", args:
6. Delete GPT Agent: "delete_agent", args: "key": "<key>"
7. Clone Repository: "clone_repository", args: "repository_url": "<url>", "clone_path": "<directory>"
8. Write to file: "write_to_file", args: "file": "<file>", "text": "<text>"
9. Read file: "read_file", args: "file": "<file>"
10. Append to file: "append_to_file", args: "file": "<file>", "text": "<text>"
11. Delete file: "delete_file", args: "file": "<file>"
12. Search Files: "search_files", args: "directory": "<directory>"
13. Analyze Code: "analyze_code", args: "code": "<full_code_string>"
14. Get Improved Code: "improve_code", args: "suggestions": "<list_of_suggestions>", "code": "<full_code_string>"
15. Write Tests: "write_tests", args: "code": "<full_code_string>", "focus": "<list_of_focus_areas>"
16. Execute Python File: "execute_python_file", args: "file": "<file>"
17. Generate Image: "generate_image", args: "prompt": "<prompt>"
18. Send Tweet: "send_tweet", args: "text": "<text>"
19. Do Nothing: "do_nothing", args:
20. Task Complete (Shutdown): "task_complete", args: "reason": "<reason>"

Составь небольшое задание для помощника, которое потребует последовательного выполнения двух команд из этого списка. 
Постарайся, чтобы команды были связаны между собой по смыслу, и чтобы задание было понятно даже новичку.
"""
    response = generate_response(prompt)
    print(f"Generated Prompt:\n{response}\n")
    return response

    
def main():
    if not os.path.exists("data"):
        os.makedirs("data")

    for i in range(10): 
        print(f"Generating data point {i+1}")
        
        prompt = generate_prompt() 
        response = generate_response(prompt) 
        
        with open(f"data/prompt_{i+1}.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        with open(f"data/response_{i+1}.txt", "w", encoding="utf-8") as f:
            f.write(response)

if __name__ == "__main__":
    main()