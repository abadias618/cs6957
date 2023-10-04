import torch
from torch import nn

class Parser(torch.nn.Module):
    def __init__(self, embeds, embedding_dim, actions_dim):
        super(Parser, self).__init__()
        
        self.embedder = embeds
        self.linear1 = nn.Linear(embedding_dim, 200)
        self.linear2 = nn.Linear(200, actions_dim)

        self.activation_function1 = nn.ReLU()
        self.activation_function2 = nn.Softmax(dim=-1)

    def forward(self, inputs):
        #out = self.linear1(inputs)
        #out = self.activation_function1(out)    
        #out = self.linear2(out)
        #out = self.activation_function2(out)
        out = self.activation_function1(inputs)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out
    
    def get_emdeddings(self, pos_embedder, glove_embedder, text, postags):
        word = torch.Tensor()
        return self.embeddings(word).view(1,-1)
