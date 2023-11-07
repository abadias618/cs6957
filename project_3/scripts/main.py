import pickle
import os
from statistics import mean
from utils import *
from n_gram import *
from lstm import *

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(1419615)

#print(os.getcwd())
with open("./data/vocab.pkl","rb") as f:
    vocab = pickle.load(f)

#print(vocab)

files = get_files("./data/dev/")
#print("seq",convert_line2idx("hello world",vocab))
pad_index = vocab['[PAD]']
print(files[1:2])
data = convert_files2idx(files[1:2], vocab)
#print(data[:3])
print("len() data",len(data))
##lstm test
#print("seq",convert_line2idx("hello world this is wonderful!!",vocab))
#train, target = make_sub_seqs(5, [convert_line2idx("hello world this is wonderful!!",vocab)], pad_index)
k=500
train, target = make_sub_seqs(k, data, pad_index)
print(len(train),len(target))
dataset = TensorDataset(torch.tensor(train), torch.tensor(target))
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)



EMBEDDING_DIM = 50
HIDDEN_DIM = 200
VOCAB_LEN = 386
TARGETS_LEN = 386
model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_LEN, TARGETS_LEN, num_layers=2)
print("model\n",model)
loss_weights = get_loss_weights(vocab, train)
print("len() loss weights",len(loss_weights))
loss_function = nn.CrossEntropyLoss(weight=torch.tensor(loss_weights), ignore_index=384, reduce="none")

optimizer = optim.Adam(model.parameters(), lr=0.1)

#with torch.no_grad():
#    inputs = train[0]
#    tag_scores = model(torch.tensor(inputs))
#    print(tag_scores)

for epoch in range(1):
    hidden = torch.zeros(2,HIDDEN_DIM)
    print("h_0",hidden.size())
    context = torch.zeros(2,HIDDEN_DIM)
    print("c_0",context.size())
    for seq, targets in dataloader:
        #print("len() seq",seq.size(),"len() targets",targets.size())
        model.zero_grad()
        tag_scores, (hidden, context) = model(seq, (hidden, context))
        #print("squeez",targets.squeeze().size())
        loss = loss_function(tag_scores, targets.squeeze())
        #print("loss",loss)
        #print("tag_scores",tag_scores.size(), hidden.size(),context.size())
        #loss = perplexity(loss)
        #loss.backward()
        optimizer.step()
print("parameters",sum(p.numel() for p in model.parameters()))


##ngram test
#seq = make_ngram_seq(4, convert_line2idx("hello world",vocab), pad_index)
#print("seq",seq)
#probs = simple_ngram_probs(seq, len(vocab))
#print("probs:",simple_ngram_probs(seq, len(vocab)))
#perp = perplexity(probs, len(seq))
#print("perp", perp)

##ngram implementation
perplexities = []
params = 0
for row in data:
    params += len(row)
    seq = make_ngram_seq(4, row, vocab)
    probs = simple_ngram_probs(seq, len(vocab))
    ppl = perplexity(probs, len(seq))
    perplexities.append(ppl)

print("\naverage perplexity over all sentences in corpora", mean(perplexities))
print("\nnumebr of params",params)