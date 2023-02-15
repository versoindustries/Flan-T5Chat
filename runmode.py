import os
import concurrent.futures
import sqlite3
import numpy as np
import time
import matplotlib as plt
import dill
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
import contextlib
import multiprocessing
import gc
from database import connect_db

# Load model from Hugging Face Hub
MODEL_SAVE_LOCATION = "./"
MODEL_NAME = "google/flan-t5-xxl"
tokenizer_path = os.path.join(MODEL_SAVE_LOCATION, "tokenizer_save")
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, cache_dir="./")
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir=MODEL_SAVE_LOCATION)
device = torch.device("cpu")
model.to(device)

# Set offload folder for tensors, only useful for GPU configuration
torch.backends.cudnn.offload_folder = "./offload"


def run_inference(input_text, previous_context, tokenizer, model, device):
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids
    start_time = time.time()
    outputs = model.generate(input_ids, max_new_tokens=1e30)
    generated_text = tokenizer.decode(outputs[0])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Generated text:")
    print(generated_text)
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))
    return generated_text, previous_context

def update_output(future, input_text, previous_context, conn):
    generated_text, previous_context = future
    conn = sqlite3.connect("input_history.db")
    conn.execute("INSERT INTO inputs_outputs (input_text, output_text) VALUES (?, ?)", (input_text, generated_text))
    conn.commit()
    return generated_text, previous_context

conn = connect_db(tokenizer, model, device)

