import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")

def generate_response(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt")
          
    outputs = model.generate(**input_ids, max_new_tokens=512)  # измените значение при необходимости

    
    response = tokenizer.decode(outputs[0])
    return response

def extract_json_from_response(response):
    start_index = response.find('{')
    end_index = response.rfind('}') + 1
    json_str = response[start_index:end_index]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def main():
  dataset = []
  for i in range(10):  # change this value to generate more data
    print(f"Generating data point {i+1}")
    input_text = "You are DataBot, an AI assistant that creates datasets. Your response must be in JSON format according to the provided schema.  \
    Response Format: \
    ```json \
    { \
        \"thoughts\": { \
            \"text\": \"thought\", \
            \"reasoning\": \"reasoning\", \
            \"plan\": \"- short bulleted\\n- list that conveys\\n- long-term plan\", \
            \"criticism\": \"constructive self-criticism\", \
            \"speak\": \"thoughts summary to say to user\" \
        }, \
        \"command\": { \
            \"name\": \"command name\", \
            \"args\": { \
                \"arg name\": \"value\" \
            } \
        } \
    } \
    ``` \
    User: Создай мне датасет с рецептами тортов, содержащий название торта, список ингредиентов и пошаговые инструкции по приготовлению."
    response = generate_response(input_text)
    json_response = extract_json_from_response(response)
    if json_response is not None:
        dataset.append({
            "prompt": input_text,
            "response": json_response
        })
    else:
      print(f"Skipping data point {i+1} due to invalid JSON")

  # Save the dataset
  with open("dataset.json", "w", encoding="utf-8") as f:
      json.dump(dataset, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()