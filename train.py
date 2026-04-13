from datasets import load_dataset  # Import the function to load datasets from Hugging Face

DATASETS=("Sp1786/multiclass-sentiment-analysis-dataset")

dataset = load_dataset(DATASETS)
dataset = dataset.map(lambda x: {'label': int(x['label'])})
dataset = dataset.filter(lambda x: isinstance(x['text'], str) and len(x['text']) > 0)

    
PRETRAINED_MODEL = "distilbert-base-uncased"  # Store the model name as a variable so we can reuse it

from transformers import AutoTokenizer  # Import the tokenizer class

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)  # Load DistilBERT's specific tokenizer

sample = "the food was not good"  # A test sentence to see what tokenization looks like
token = tokenizer(sample)  # Convert the sample sentence into token IDs
print("token: ", token)  # Print the tokenized output to inspect it

def tokenize_function(examples):  # Define a function to tokenize batches of examples
    return tokenizer(examples['text'], truncation=True)  # Tokenize the 'sentence' column, cut if too long

from transformers import DataCollatorWithPadding  # Import collator that pads sequences dynamically per batch

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # Create the collator using our tokenizer

tokenized_dataset = dataset.map(tokenize_function, batched=True)  # Apply tokenize_function to every example in the dataset
print("tokenized_dataset: ", tokenized_dataset)  # Print the dataset structure to confirm tokenization worked

from transformers import AutoModelForSequenceClassification  # Import the model class for classification tasks

model = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL,  # Load DistilBERT base weights
    num_labels=3  # Attach a classification head with 2 outputs: positive and negative
)

from transformers import TrainingArguments  # Import the class that holds all training configuration

training_args = TrainingArguments(
    output_dir="./results",  # Save model checkpoints here during training
    eval_strategy="epoch",  # Evaluate on validation set after every epoch
    num_train_epochs=3,  # Loop through the training data 3 times
    per_device_train_batch_size=16,  # Process 16 examples at a time during training
    per_device_eval_batch_size=16,  # Process 16 examples at a time during evaluation
    logging_dir="./logs",  # Save training logs here
)

import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

from transformers import Trainer  # Import the Trainer class that handles the training loop

trainer = Trainer(
    model=model,  # The model to train
    args=training_args,  # The training configuration we defined above
    train_dataset=tokenized_dataset["train"].select(range(4000)),  # Use only 4000 examples for training
    eval_dataset=tokenized_dataset["validation"].select(range(500)),  # Use only 500 examples for validation
    # train_dataset=tokenized_dataset["train"],  # Use only 4000 examples for training
    # eval_dataset=tokenized_dataset["validation"],  # Use only 500 examples for validation
    data_collator=data_collator,  # Use dynamic padding per batch
    compute_metrics=compute_metrics,
)

trainer.train()
# Note: trainer.train() is missing here — this is where training actually starts

model.save_pretrained("./sentiment_model_3class")
tokenizer.save_pretrained("./sentiment_model_3class")

