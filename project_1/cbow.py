import sys
sys.path.insert(0, './scripts/')
from torch import nn
import torch

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
    word = torch.LongTensor([vocab[word]])
    return self.embeddings(word).view(1,-1)

def make_sentence_vector(sentence, word_dict):
    idxs = [word_dict[w] for w in sentence]
    return torch.tensor(idxs, dtype=torch.long)

