import sqlite3
import time
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_SAVE_LOCATION = "./"
MODEL_NAME = "google/flan-t5-xxl"
tokenizer_path = os.path.join(MODEL_SAVE_LOCATION, "tokenizer_save")
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, cache_dir="./")
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir=MODEL_SAVE_LOCATION)
device = torch.device("cpu")
model.to(device)

def connect_db(tokenizer, model, device):
    conn = sqlite3.connect("input_history.db")
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS inputs_outputs (input_text TEXT, output_text TEXT)")
    except sqlite3.OperationalError as e:
        if "table inputs_outputs has no column named input_text" in str(e):
            conn.execute("ALTER TABLE inputs_outputs ADD COLUMN input_text TEXT")
        else:
            raise e
    return conn

conn = connect_db(tokenizer, model, device)

# Code to handle user input
previous_context = ""
while True:
        from runmode import tokenizer, model, device, run_inference, update_output
        input_text = input("Enter your message: ")
        future = run_inference(input_text, previous_context, tokenizer, model, device)
        output, previous_context = update_output(future, input_text, previous_context, conn)
        print("Output:", output)
        input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids

        start_time = time.time()
        outputs = model.generate(input_ids, max_new_tokens=1e30)
        generated_text = tokenizer.decode(outputs[0])
        end_time = time.time()
        
        conn.execute("INSERT INTO inputs_outputs (input_text, output_text) VALUES (?, ?)", (input_text, generated_text))
        conn.commit()

        elapsed_time = end_time - start_time
        print("Generated text:")
        print(generated_text)
        print("Elapsed time: {:.2f} seconds".format(elapsed_time))

        conn.close()

