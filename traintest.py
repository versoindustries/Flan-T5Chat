from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

# Load dataset
dataset_id = "soumith/samsum"
dataset = load_dataset(dataset_id)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Preprocess data
def preprocess_function(sample, padding="max_length"):
    # Add prefix to the input for T5
    inputs = ["summarize: " + item for item in sample["dialogue"]]

    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=512, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["summary"], max_length=128, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])

# Prepare training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=2,
    num_train_epochs=3,
    logging_steps=100,
    learning_rate=3e-4,
    warmup_steps=500,
    save_strategy="epoch",
    report_to="wandb",
    run_name="flan-t5-samsum"
)

# Prepare trainer
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=lambda preds, labels: evaluate_metric(preds, labels),
)

# Helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

# Evaluate function
def evaluate_metric(preds, labels):
    preds = np.argmax(preds.predictions, axis=-1)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
result = evaluate.evaluate(decoded_preds, decoded_labels)
return result

trainer.train()

eval_result = trainer.evaluate()
print(eval_result)
