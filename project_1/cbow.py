import sys
sys.path.insert(0, './scripts/')
from torch import nn
import torch

class CBOW(torch.nn.Module):
    def __init__(self, corpus_length, embedding_dim):
        super(CBOW, self).__init__()
        
        self.embeddings = nn.Embedding(corpus_length,embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, corpus_length)
        self.activation_function = nn.LogSoftmax(dim = -1)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).sum(1).squeeze(1)#sum(self.embeddings(inputs)).view(1,-1)
        out = self.linear1(embeds)
        out = self.activation_function(out)
        return out
    
    def get_word_emdedding(self, vocab, word):
        word = torch.LongTensor([vocab[word]])
        return self.embeddings(word).view(1,-1)


