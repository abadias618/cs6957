from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, num_layers = 1, transition_dim = 64):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=False)
        # The linear layer that maps from hidden state space to tag space
        self.linear1 = nn.Linear(hidden_dim, transition_dim)
        self.hidden2tag = nn.Linear(transition_dim, tagset_size)

        self.activation_function1 = nn.ReLU()
        self.activation_function2 = nn.Softmax(dim=-1)

    def forward(self, sentence, h_n_c):
        embeds = self.word_embeddings(sentence)
        #print("embeds",embeds.size())
        out, (hidden, context) = self.lstm(embeds.squeeze(), h_n_c)
        #print("out, hidden, context",out.size(), hidden.size(), context.size())
        out = self.linear1(out)
        out = self.activation_function1(out)
        tag_space = self.hidden2tag(out)
        #print("tag_space",tag_space.size())
        tag_scores = self.activation_function2(tag_space)
        #print("tag_scores",tag_scores.size())

        return tag_scores, (hidden, context)
    


def get_loss_weights(vocab, training_data):
    stream = []
    for l in training_data:
        stream += l

    w = []
    counts = Counter(stream)
    for i in range(len(vocab)):
        w.append(1-(counts[i]/sum([c for c in counts.values()])))
    return w