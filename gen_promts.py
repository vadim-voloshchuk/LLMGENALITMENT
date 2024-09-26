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
    prompt = """You are a senior developer, and you are giving a task to your assistant who is just learning to program. 
Your assistant can execute the following commands:

1. "google": Search for something on Google.
    Example: "google", args: {"input": "best pizza recipes"}

2. "browse_website": Find information on a website.
    Example: "browse_website", args: {"url": "https://wikipedia.org", "question": "When was Google founded?"}

3. "write_to_file": Write text to a file.
    Example: "write_to_file", args: {"file": "answer.txt", "text": "Hello, world!"}

4. "read_file": Read text from a file.
    Example: "read_file", args: {"file": "input.txt"}

Create a short task for your assistant that requires the sequential execution of two commands from this list. 
Try to make the commands logically connected and the task clear even for a beginner.

For example: 
1. Search Google for "highest mountains in the world".
2. Write the search results to a file "mountains.txt".
"""
    response = generate_response(prompt)
    print(f"Generated Prompt:\n{response}\n")
    return response
    
def main():
    count = int(input("СКОЛЬКО: "))
    if not os.path.exists("data"):
        os.makedirs("data")

    for i in range(count): 
        print(f"Generating data point {i+1}")
        
        prompt = generate_prompt() 
        response = generate_response(prompt) 
        
        with open(f"data/prompt_{i+1}.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        with open(f"data/response_{i+1}.txt", "w", encoding="utf-8") as f:
            f.write(response)

if __name__ == "__main__":
    main()