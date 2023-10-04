from scripts import dataloader
from scripts import state
import numpy as np
import random
import torch
from torch import nn
import torchtext

torch.manual_seed(1419615)
random.seed(1419615)
np.random.seed(1419615)

def main():
    #load data
    complete_data = dataloader.load_data("./data/train.txt")
    #print(complete_data[:2],"\n")
    hidden_data = dataloader.load_hidden("./data/hidden.txt")
    #print(hidden_data[:2],"\n")
    pos_set, pos_set_idx2name, pos_set_name2idx = dataloader.load_pos_set("./data/pos_set.txt")
    #print(pos_set,"\n")
    tagset, tag_set_idx2name, tag_set_name2idx = dataloader.load_tagset("./data/tagset.txt")
    #print(tagset,"\n")

    data = [] #tokens-dependencies-ParseState
    #put data into objs
    for row in complete_data[:2]:
        tokens = \
        [state.Token(i+1,input_token,pos_tag) for i, (input_token, pos_tag) in enumerate(zip(row[0], row[1]))]
        data.append([tokens, row[2]])
    #print("data sanity check\n\n",data)
    C_WINDOW = 2
    NUMBER_OF_POSTAGS = len(pos_set)
    DIM = 50
    # Glove embeddings
    glove = torchtext.vocab.GloVe(name="6B", dim=DIM)
    # Torch embeddings
    torch_emb = nn.Embedding(NUMBER_OF_POSTAGS, DIM)
    train = []
    for row in data:
        s = state.ParseState([],row[0],[])
        #print("row[1] sanity check\n\n", row[1])
        for action in row[1]:
            #print("action\n",action)
            if action not in tagset:
                raise Exception()
            
            a = action.split("_")
            
            if  len(a) > 1:
                if a[1] == "L":
                    state.left_arc(s, action)
                elif a[1] == "R":
                    state.right_arc(s, action)
            else:
                state.shift(s)

            #pad
            w_stack = [w.word for w in s.stack]
            w_stack = state.pad(w_stack, C_WINDOW, "token")
            p_stack = [p.pos for p in s.stack]
            p_stack = state.pad(p_stack, C_WINDOW, "postag")

            w_buffer = [w.word for w in s.parse_buffer]
            w_buffer = state.pad(w_buffer, C_WINDOW, "token")
            p_buffer = [p.pos for p in s.parse_buffer]
            p_buffer = state.pad(p_buffer, C_WINDOW, "postag")

            w = w_stack + w_buffer
            print("\nw",w)
            w_emb = glove.get_vecs_by_tokens(w, lower_case_backup=True)
            print("\nw_emb",w_emb.size())
            # mean representation
            w_emb_mean = torch.mean(w_emb, 0)
            print("\nw_emb mean",w_emb_mean.size())
            # concat representation
            w_emb_concat = w_emb[0]
            for i in range(1,len(w_emb)-1):
                w_emb_concat = torch.cat((w_emb_concat, w_emb[i]),0)
            
            print("\nw_emb cat",w_emb_concat.size())

            
            p = p_stack + p_buffer
            print("\np",p)
            p = [pos_set_name2idx[tag] for tag in p]
            print("\np num",p)
            p_emb = torch_emb(torch.Tensor(p).to(torch.int64))
            print("\np_emb",p_emb.size())
            # mean representation
            p_emb_mean = torch.mean(p_emb, 0)
            print("\np_emb mean",p_emb_mean.size())
            # concat representation
            p_emb = torch.flatten(p_emb, start_dim=1)
            print("\nFlat",p_emb.size())
            p_emb_concat = p_emb[0]
            for i in range(1,len(p_emb)-1):
                p_emb_concat = torch.cat((p_emb_concat, p_emb[i]),0)
            print("\np_emb cat",p_emb_concat.size())

            # put vecs together
            print("\nmash",torch.add(w_emb_mean, p_emb_mean).size())
            train.append(None)
    print("FINAL\n\n")
        
main()