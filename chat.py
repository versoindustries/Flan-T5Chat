import os
import concurrent.futures
import sqlite3
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model from Hugging Face Hub
MODEL_SAVE_LOCATION = "./"
MODEL_NAME = "google/flan-t5-xxl"
tokenizer_path = os.path.join(MODEL_SAVE_LOCATION, "tokenizer_save")
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, cache_dir="./")
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir=MODEL_SAVE_LOCATION)
device = torch.device("cpu")
model.to(device)

# Connect to database for storing inputs and outputs
conn = sqlite3.connect("input_history.db")
conn.execute("CREATE TABLE IF NOT EXISTS inputs_outputs (input_text TEXT, output_text TEXT)")
conn.commit()

# Define a function to preprocess the input text and previous context
def preprocess_input(input_text, previous_context):
    # Add special tokens for task and context
    input_text = "<flan:generate>" + input_text + "</s>"
    if previous_context:
        input_text = "<flan:context>" + previous_context + "</flan:context>" + input_text
    return input_text

# Define a function to postprocess the output text and update the previous context
def postprocess_output(output_text, previous_context):
    # Remove special tokens from output
    output_text = output_text.replace("<pad>", "").strip()
    # Update previous context with output
    if previous_context:
        previous_context += "\n" + output_text
    else:
        previous_context = output_text
    return output_text, previous_context

# Define a function to run inference on some text using the T5 model
def run_inference(input_text, previous_context):
    # Convert input text to tensor of ids using tokenizer 
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True).input_ids
    
    # Generate output text using model 
    outputs = model.generate(input_ids)
    
    # Convert output ids to text using tokenizer 
    generated_text = tokenizer.decode(outputs[0])
    
    # Postprocess the output text and update the previous context 
    generated_text, previous_context = postprocess_output(generated_text, previous_context)
    
    return generated_text, previous_context

# Define an executor for running inference in parallel 
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Initialize an empty string for storing previous context 
previous_context = ""

# Loop for getting user input and generating output 
while True:
    
     # Get user input text 
     input_text = input("Enter some text: ")
     
     # Preprocess the input text and previous context 
     input_text = preprocess_input(input_text, previous_context)
     
     # Run inference using executor and get future object 
     future = executor.submit(run_inference, input_text, previous_context)
     
     # Wait for future to finish and get result 
     generated_text, previous_context = future.result()
     
     # Print generated text 
     print("Generated text:")
     print(generated_text)
