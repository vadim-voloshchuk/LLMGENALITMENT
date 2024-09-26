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
    prompt = """Ты — старший разработчик, и даешь задание своему помощнику, который только учится программировать. 
Помощник умеет выполнять следующие команды:

1. "google": найти что-то в Google. 
    Пример: "google", args: {"input": "лучшие рецепты пиццы"}

2. "browse_website": найти информацию на сайте.  
    Пример: "browse_website", args: {"url": "https://wikipedia.org", "question": "Когда основали Google?"}

3. "write_to_file": записать текст в файл. 
    Пример: "write_to_file", args: {"file": "ответ.txt", "text": "Привет, мир!"}

4. "read_file": прочитать текст из файла.
    Пример: "read_file", args: {"file": "input.txt"}

Составь небольшое задание для помощника, которое потребует последовательного выполнения двух команд из этого списка. 
Постарайся, чтобы команды были связаны между собой по смыслу, и чтобы задание было понятно даже новичку.

Например: 
1. Найди в Google "самые высокие горы мира". 
2. Результат поиска запиши в файл "горы.txt".
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