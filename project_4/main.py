from utils import *
import torch
import torch.nn as nn
torch.manual_seed(1419615)
from datasets import load_dataset, load_metric
from datasets import load_dataset_builder
#dataset_rte = load_dataset_builder("yangwang825/rte")
#dataset_sst2 = load_dataset_builder("gpt3mix/sst2")
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, BertModel, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
#from transformers import pipeline
import evaluate
import numpy as np

rte_train = load_dataset("yangwang825/rte", split="train").select(range(10))#, streaming=True)
rte_test = load_dataset("yangwang825/rte", split="test").select(range(10))#, streaming=True)
sst2_train = load_dataset("gpt3mix/sst2", split="train").select(range(10))#, streaming=True)
sst2_test = load_dataset("gpt3mix/sst2", split="test").select(range(10))#, streaming=True)

#print(rte_dataset)
#print(sst2_dataset)

tiny = "prajjwal1/bert-tiny"
mini = "prajjwal1/bert-mini"
tiny_tokenizer = AutoTokenizer.from_pretrained(tiny)
tiny_model = AutoModelForSequenceClassification.from_pretrained(tiny)
mini_tokenizer = AutoTokenizer.from_pretrained(mini)
mini_model = BertModel.from_pretrained(mini)

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


enc_rte_train = enc_rte_train.remove_columns(["text1","text2","idx","label_text"])
#enc_rte_train = enc_rte_train.rename_column("label","labels")
enc_rte_train.set_format("torch")
print("\ntest data\n",enc_rte_train[1])
print("cols",enc_rte_train.column_names)
enc_rte_test = enc_rte_test.remove_columns(["text1","text2","idx","label_text"])
#enc_rte_test = enc_rte_test.rename_column("label","labels")
enc_rte_test.set_format("torch")
#print("enc_rte_train[1]",enc_rte_train[1])

tiny_data_collator = DataCollatorWithPadding(tokenizer=tiny_tokenizer)
mini_data_collator = DataCollatorWithPadding(tokenizer=mini_tokenizer)

from torch.utils.data import DataLoader

rte_train_dataloader = DataLoader(
    enc_rte_train, shuffle=True, batch_size=8, collate_fn=tiny_data_collator
)
rte_test_dataloader = DataLoader(
    enc_rte_test, batch_size=8, collate_fn=tiny_data_collator
)

from transformers import AdamW

optimizer = AdamW(tiny_model.parameters(), lr=1e-4)
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(rte_train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print("num_training_steps",num_training_steps)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tiny_model.to(device)
print("device",device)

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

tiny_model.train()
for epoch in range(num_epochs):
    for batch in rte_train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        print("batch",batch)
        outputs = tiny_model(**batch)
        print("outputs",outputs)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)




accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

BATCH_SIZE = 8
LR = [1e-4, 1e-5, 1e-6]
OUTPUT_DIR = "/scratch/general/vast/u1419615/cs6957/assignment4/models"
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
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
    train_dataset=rte_train,
    eval_dataset=rte_test,
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