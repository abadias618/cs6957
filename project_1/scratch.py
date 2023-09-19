import torch
from torch import nn
import numpy as np


text = ""
text = text.replace(',','').replace('.','').lower().split()
corpus = set(text)
corpus_length = len(corpus)
word_dict = {}
inverse_word_dict = {}
for i, word in enumerate(corpus):
    word_dict[word] = i
    inverse_word_dict[i] = word
data = []
for i in range(2, len(text) - 2):
    sentence = [text[i-2], text[i-1],
                text[i+1], text[i+2]]
    target = text[i]
    data.append((sentence, target)) 
print(data[3])

embedding_length = 20

class CBOW(torch.nn.Module):
    def __init__(self, corpus_length, embedding_dim):
        super(CBOW, self).__init__()

        self.embeddings = nn.Embedding(corpus_length,embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 64)
        self.linear2 = nn.Linear(64, corpus_length)

        self.activation_function1 = nn.ReLU()
        self.activation_function2 = nn.LogSoftmax(dim = -1)
    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1,-1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

def get_word_emdedding(self, word):
    word = torch.LongTensor([word_dict[word]])
    return self.embeddings(word).view(1,-1)

model = CBOW(corpus_length, embedding_length)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def make_sentence_vector(sentence, word_dict):
    idxs = [word_dict[w] for w in sentence]
    return torch.tensor(idxs, dtype=torch.long)
print(make_sentence_vector(['stormy','nights','when','the'], word_dict))

for epoch in range(100):
    epoch_loss = 0
    for sentence, target in data:
        model.zero_grad()
        sentence_vector = make_sentence_vector(sentence, word_dict)
        log_probs = model(sentence_vector)
        loss = loss_function(log_probs, torch.tensor(
        [word_dict[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.data
    print('Epoch: '+str(epoch)+', Loss: ' + str(epoch_loss.item()))

def get_predicted_result(input, inverse_word_dict):
    index = np.argmax(input)
    return inverse_word_dict[index]

def predict_sentence(sentence):
    sentence_split = sentence.replace('.','').lower().split()
    sentence_vector = make_sentence_vector(sentence_split, word_dict)
    prediction_array = model(sentence_vector).data.numpy()
    print('Preceding Words: {}\n'.format(sentence_split[:2]))
    print('Predicted Word: {}\n'.format(get_predicted_result(prediction_array[0],inverse_word_dict)))
    print('Following Words: {}\n'.format(sentence_split[2:]))
predict_sentence('to see leap and')