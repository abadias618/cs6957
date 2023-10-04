import torch
from torch import nn

class Parser(torch.nn.Module):
    def __init__(self, embeds, embedding_dim, actions):
        super(Parser, self).__init__()
        
        self.embedder = embeds
        self.linear1 = nn.Linear(embedding_dim, 200)
        self.linear2 = nn.Linear(200, actions)

        self.activation_function1 = nn.ReLU()
        self.activation_function2 = nn.Softmax()

    def forward(self, inputs):
        out = self.linear1(inputs)
        out = self.activation_function1(out)    
        out = self.linear2(out)
        out = self.activation_function2(out)
    
    def get_word_emdedding(self, vocab, word):
        word = torch.LongTensor([vocab[word]])
        return self.embeddings(word).view(1,-1)
