from utils import *
import torch
import torch.nn as nn
torch.manual_seed(1419615)
from datasets import load_dataset, load_metric
from datasets import load_dataset_builder
#dataset_rte = load_dataset_builder("yangwang825/rte")
#dataset_sst2 = load_dataset_builder("gpt3mix/sst2")
from transformers import AutoTokenizer
from transformers import AutoModel, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
#from transformers import pipeline
import evaluate
import numpy as np

rte_train = load_dataset("yangwang825/rte", split="train", streaming=True)
rte_test = load_dataset("yangwang825/rte", split="test", streaming=True)
sst2_train = load_dataset("gpt3mix/sst2", split="train", streaming=True)
sst2_test = load_dataset("gpt3mix/sst2", split="test", streaming=True)

#print(rte_dataset)
#print(sst2_dataset)

tiny = "prajjwal1/bert-tiny"
mini = "prajjwal1/bert-mini"
tiny_tokenizer = AutoTokenizer.from_pretrained(tiny)
tiny_model = AutoModel.from_pretrained(tiny)
mini_tokenizer = AutoTokenizer.from_pretrained(mini)
mini_model = AutoModel.from_pretrained(tiny)

def preprocess_rte(examples, model_label="tiny"):
  if model_label == "tiny":
    return tiny_tokenizer(examples["text1"], examples["text2"], truncation=True, padding=True, return_tensors="pt")
  else:
    return mini_tokenizer(examples["text1"], examples["text2"], truncation=True, padding=True, return_tensors="pt")

def preprocess_sst2(examples, model_label="tiny"):
  if model_label == "tiny":
    return tiny_tokenizer(examples["text"], truncation=True, padding=True, return_tensors="pt")
  else:
    return mini_tokenizer(examples["text"], truncation=True, padding=True, return_tensors="pt")

enc_rte_train = rte_train.map(preprocess_rte, batched=True)
enc_rte_test = rte_test.map(preprocess_rte, batched=True)
enc_sst2_train = sst2_train.map(preprocess_sst2, batched=True)
enc_sst2_test = sst2_test.map(preprocess_sst2, batched=True)

tiny_data_collator = DataCollatorWithPadding(tokenizer=tiny_tokenizer)
mini_data_collator = DataCollatorWithPadding(tokenizer=mini_tokenizer)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

BATCH_SIZE = 8
LR = [1e-4, 1e-5, 1e-6]
training_args = TrainingArguments(
    output_dir="",
    learning_rate=LR[0],
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

tiny_rte_trainer = Trainer(
    model=tiny_model,
    args=training_args,
    train_dataset=enc_rte_train,
    eval_dataset=enc_rte_test,
    tokenizer=tiny_tokenizer,
    data_collator=tiny_data_collator,
    compute_metrics=compute_metrics,
)
print("Started training tiny rte\n")
tiny_rte_trainer.train()
print("Finished training tiny rte\n")

mini_rte_trainer = Trainer(
    model=mini_model,
    args=training_args,
    train_dataset=enc_rte_train,
    eval_dataset=enc_rte_test,
    tokenizer=mini_tokenizer,
    data_collator=mini_data_collator,
    compute_metrics=compute_metrics,
)
print("Started training mini rte\n")
mini_rte_trainer.train()
print("Finished training mini rte\n")


x = load_rte_data("./hidden_rte.csv")
print(x[:1])
y = load_sst2_data("./hidden_sst2.csv")
print(y[:1])

tokenized_inp = tiny_tokenizer(y[0], truncation=True, padding=True, return_tensors="pt")
with torch.no_grad():
    out = tiny_model(**tokenized_inp)
    print("out\n",out.pooler_output)

logits = nn.Linear(128,2)(out.pooler_output)
print("logits\n",logits)

#enc = tiny_tokenizer.encode(x[0])
#for tok in enc:
#    print(tok, tiny_tokenizer.decode(tok))
#rte_pipeline = pipeline("rte", model = tiny_model, tokenizer = tiny_tokenizer)
#rte_pipeline(
#    [
#        "I've been waiting for a HuggingFace course my whole life.",
#        "I hate this so much!",
#    ]
#)