from scripts import dataloader
from scripts import state
import torch
from torch import nn
import torchtext


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
    # Glove embeddings
    glove = torchtext.vocab.GloVe(name="6B", dim=50)
    # Torch embeddings
    torch_emb = nn.Embedding(C_WINDOW*2, 50)
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
            print("\nw_emb mean",w_emb.mean())
            # concat representation
            
            p = p_stack + p_buffer
            print("\np",p)
            p = [pos_set_name2idx[tag] for tag in p]
            print("\np num",p)
            p_emb = torch_emb(torch.Tensor(p))
            print("\np_emb",p_emb.size())
            print("\np_emb mean",p_emb.mean())
            
    print("FINAL\n\n")
        
main()