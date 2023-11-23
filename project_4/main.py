from utils import *
import torch
import torch.nn as nn
torch.manual_seed(1419615)
from datasets import load_dataset, load_metric
from datasets import load_dataset_builder
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, BertModel, TrainingArguments, Trainer, BertModel
from transformers import DataCollatorWithPadding
from transformers import pipeline
import evaluate
import numpy as np

rte_train = load_dataset("yangwang825/rte", split="train")
rte_test = load_dataset("yangwang825/rte", split="test")
sst2_train = load_dataset("gpt3mix/sst2", split="train")
sst2_test = load_dataset("gpt3mix/sst2", split="test")

print("Loaded Datasets...")
#print(sst2_dataset)

tiny = "prajjwal1/bert-tiny"
mini = "prajjwal1/bert-mini"
tiny_tokenizer = AutoTokenizer.from_pretrained(tiny)
tiny_model = AutoModelForSequenceClassification.from_pretrained(tiny)
mini_tokenizer = AutoTokenizer.from_pretrained(mini)
mini_model = AutoModelForSequenceClassification.from_pretrained(mini)

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

print("Preprocessing...Done.")

enc_rte_train = enc_rte_train.remove_columns(["text1","text2","idx","label_text"])
enc_rte_train.set_format("torch")
enc_rte_test = enc_rte_test.remove_columns(["text1","text2","idx","label_text"])
enc_rte_test.set_format("torch")

enc_sst2_train = enc_sst2_train.remove_columns(["text"])
enc_sst2_train.set_format("torch")
enc_sst2_test = enc_sst2_test.remove_columns(["text"])
enc_sst2_test.set_format("torch")

tiny_data_collator = DataCollatorWithPadding(tokenizer=tiny_tokenizer)
mini_data_collator = DataCollatorWithPadding(tokenizer=mini_tokenizer)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tiny_model.to(device)
mini_model.to(device)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# BASELINE MODELS

rte_baseline = np.array(get_baseline(len(enc_rte_test)))
z_rte = np.zeros_like(rte_baseline)
sst2_baseline = get_baseline(len(enc_sst2_test))
z_sst2 = np.zeros_like(sst2_baseline)
print("Baselines")
print(compute_metrics((np.array([rte_baseline,z_rte]).T, enc_rte_test["label"])))
print(compute_metrics((np.array([sst2_baseline, z_sst2]).T, enc_sst2_test["label"])))


EPOCHS = 10
BATCH_SIZE = 8
LR = [1e-4, 1e-5, 1e-6]
OUTPUT_DIR = "/scratch/general/vast/u1419615/cs6957/assignment4/models"
for lr in LR:
  print(f"\nFor Learning Rate: {lr}.")
  training_args = TrainingArguments(
      output_dir=OUTPUT_DIR,
      learning_rate=LR[0],
      per_device_train_batch_size=BATCH_SIZE,
      per_device_eval_batch_size=BATCH_SIZE,
      num_train_epochs=EPOCHS,
      evaluation_strategy="epoch",
      save_strategy="epoch",
      load_best_model_at_end=True,
  )

  # RTE TASK

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

  # SST2 TASK

  tiny_sst2_trainer = Trainer(
      model=tiny_model,
      args=training_args,
      train_dataset=enc_sst2_train,
      eval_dataset=enc_sst2_test,
      tokenizer=tiny_tokenizer,
      data_collator=tiny_data_collator,
      compute_metrics=compute_metrics,
  )
  print("Started training tiny sst2\n")
  tiny_rte_trainer.train()
  print("Finished training tiny sst2\n")

  mini_sst2_trainer = Trainer(
      model=mini_model,
      args=training_args,
      train_dataset=enc_sst2_train,
      eval_dataset=enc_sst2_test,
      tokenizer=mini_tokenizer,
      data_collator=mini_data_collator,
      compute_metrics=compute_metrics,
  )
  print("Started training mini sst2\n")
  mini_rte_trainer.train()
  print("Finished training mini sst2\n")


# NO finetunning rte (tiny) 128
f = tiny_tokenizer([x["text1"] for x in rte_test], [x["text2"] for x in rte_test],truncation=True ,return_tensors="pt", padding=True)

tiny_b_model = BertModel.from_pretrained(tiny)
with torch.no_grad():
    out = tiny_b_model(**f)

logits = nn.Linear(128,2)(out.pooler_output)
print("NO finetunning rte (tiny) acc", compute_metrics((logits.detach().numpy(), rte_test["label"])))

# NO finetunning rte (tiny) 256
f = mini_tokenizer([x["text1"] for x in rte_test], [x["text2"] for x in rte_test],truncation=True ,return_tensors="pt", padding=True)

mini_b_model = BertModel.from_pretrained(mini)
with torch.no_grad():
    out = mini_b_model(**f)

logits = nn.Linear(256,2)(out.pooler_output)
print("NO finetunning rte (mini) acc", compute_metrics((logits.detach().numpy(), rte_test["label"])))


# NO finetunning sst2 (tiny) 128
f = tiny_tokenizer.batch_encode_plus([x["text"] for x in sst2_test],truncation=True ,return_tensors="pt", padding=True, return_attention_mask=True)

tiny_b_model = BertModel.from_pretrained(tiny)
with torch.no_grad():
    out = tiny_b_model(**f)

logits = nn.Linear(128,2)(out.pooler_output)
print("NO finetunning sst2 (tiny) acc", compute_metrics((logits.detach().numpy(), sst2_test["label"])))

# NO finetunning sst2 (mini) 256
f = mini_tokenizer.batch_encode_plus([x["text"] for x in sst2_test],truncation=True ,return_tensors="pt", padding=True, return_attention_mask=True)

mini_b_model = BertModel.from_pretrained(mini)
with torch.no_grad():
    out = mini_b_model(**f)

logits = nn.Linear(256,2)(out.pooler_output)
print("NO finetunning sst2 (mini) acc", compute_metrics((logits.detach().numpy(), sst2_test["label"])))


