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

tiny = "prajjwal1/bert-tiny"
mini = "prajjwal1/bert-mini"
tiny_tokenizer = AutoTokenizer.from_pretrained(tiny)
srt_mini_model = AutoModelForSequenceClassification.from_pretrained("/scratch/general/vast/u1419615/cs6957/assignment4/models/checkpoint-1560")
mini_tokenizer = AutoTokenizer.from_pretrained(mini)
sst2_tiny_model = AutoModelForSequenceClassification.from_pretrained("/scratch/general/vast/u1419615/cs6957/assignment4/models/checkpoint-1872")

h_rte = load_rte_data("./hidden_rte.csv")
f = tiny_tokenizer([x[0] for x in h_rte], [x[1] for x in h_rte],truncation=True ,return_tensors="pt", padding=True)
with torch.no_grad():
    out = srt_mini_model(**f)
print(out)


h_sst2 = load_sst2_data("./hidden_sst2.csv")
print(h_sst2[:1])
f = mini_tokenizer([x for x in h_sst2],truncation=True ,return_tensors="pt", padding=True)
with torch.no_grad():
    out = sst2_tiny_model(**f)
print(out)
