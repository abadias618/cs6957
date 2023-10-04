from scripts import dataloader
from scripts import state
from helper import *
from Parser import *
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torchtext
from tqdm import tqdm

torch.manual_seed(1419615)
random.seed(1419615)
np.random.seed(1419615)

def main():
    #load data
    complete_data = dataloader.load_data("./data/dev.txt")
    #print(complete_data[:2],"\n")
    pos_set, pos_set_idx2name, pos_set_name2idx = dataloader.load_pos_set("./data/pos_set.txt")
    #print(pos_set,"\n")
    tagset, tag_set_idx2name, tag_set_name2idx = dataloader.load_tagset("./data/tagset.txt")
    #print(tagset,"\n")

    data = [] #tokens-dependencies-ParseState
    #put data into objs
    for row in complete_data[:100]:
        tokens = \
        [state.Token(i+1,input_token,pos_tag) for i, (input_token, pos_tag) in enumerate(zip(row[0], row[1]))]
        data.append([tokens, row[2]])
    #print("data sanity check\n\n",data)
    C_WINDOW = 2
    NUMBER_OF_POSTAGS = len(pos_set)
    NUMBER_OF_ACTIONS = len(tagset)
    DIM = 50
    # Glove embeddings
    glove = torchtext.vocab.GloVe(name="6B", dim=DIM)
    # Torch embeddings
    torch_emb = nn.Embedding(NUMBER_OF_POSTAGS, DIM)
    
    train_mean, train_concat, labels = prepare_vectors_for_training(data, tagset=tagset,
                                                       C_WINDOW=C_WINDOW, glove=glove,
                                                       torch_emb=torch_emb,
                                                       pos_set_name2idx=pos_set_name2idx,
                                                       tag_set_name2idx=tag_set_name2idx)
    #READY DATA
    dataset_mean = TensorDataset(torch.stack(train_mean), torch.tensor(labels))
    dataloader_mean = DataLoader(dataset_mean, batch_size = 64, shuffle=False)

    dataset_concat = TensorDataset(torch.stack(train_concat), torch.tensor(labels))
    dataloader_concat = DataLoader(dataset_concat, batch_size = 256, shuffle=False)
    print("FINALIZED DATA CREATION\n\n")

    
    # Create Models
    model_mean = Parser(DIM, NUMBER_OF_ACTIONS)
    print("MODEL_MEAN CREATED\n",model_mean)
    model_concat = Parser(DIM, NUMBER_OF_ACTIONS)
    print("MODEL_CONCAT CREATED\n",model_concat)
    loss_function = nn.CrossEntropyLoss()
    optimizer_mean = torch.optim.Adam(model_mean.parameters(), lr=0.001)
    optimizer_concat = torch.optim.Adam(model_concat.parameters(), lr=0.001)

    # train
    model_mean = train_model(dataloader_mean, model_mean, loss_function, optimizer_mean, epochs=1)
    #model_concat = train_model(dataloader_concat, model_concat, loss_function, optimizer_concat, epochs=1)
    
    pred = model_mean(train_mean[0])
    #print("raw pred\n",pred)
    #print("pred.data",pred.data)
    #print("argmax",np.argmax(pred.data.numpy()))
    print("result?",tag_set_idx2name[round(np.argmax(pred.data.numpy()))])

    hidden_data = dataloader.load_hidden("./data/hidden.txt")
    ##print(hidden_data[:2],"\n")
    obj_hidden_data = [] #tokens-dependencies-ParseState
    #put data into objs
    for row in hidden_data[:100]:
        tokens = \
        [state.Token(i+1,input_token,pos_tag) for i, (input_token, pos_tag) in enumerate(zip(row[0], row[1]))]
        obj_hidden_data.append(tokens)

    # loop
    for row in obj_hidden_data:
        s = state.ParseState([],row,[])
        while not state.is_final_state(s, C_WINDOW):
            w_stack = [w.word for w in s.stack]
            w_stack = state.pad(w_stack, C_WINDOW, "token")
            p_stack = [p.pos for p in s.stack]
            p_stack = state.pad(p_stack, C_WINDOW, "postag")

            w_buffer = [w.word for w in s.parse_buffer]
            w_buffer = state.pad(w_buffer, C_WINDOW, "token")
            p_buffer = [p.pos for p in s.parse_buffer]
            p_buffer = state.pad(p_buffer, C_WINDOW, "postag")

            w = w_stack + w_buffer
            w_emb = glove.get_vecs_by_tokens(w, lower_case_backup=True)
            # mean representation
            w_emb_mean = torch.mean(w_emb, 0)
            # concat representation
            w_emb_concat = w_emb[0]
            for i in range(1,len(w_emb)):
                w_emb_concat = torch.cat((w_emb_concat, w_emb[i]),0)

            
            p = p_stack + p_buffer
            p = [pos_set_name2idx[tag] for tag in p]
            p_emb = torch_emb(torch.Tensor(p).to(torch.int64))
            # mean representation
            p_emb_mean = torch.mean(p_emb, 0)
            # concat representation
            p_emb_concat = p_emb[0]
            for i in range(1,len(p_emb)):
                p_emb_concat = torch.cat((p_emb_concat, p_emb[i]),0)

            pred = model_mean(torch.add(w_emb_mean, p_emb_mean))
            pred_text = tag_set_idx2name[round(np.argmax(pred.data.numpy()))]

            action = pred_text

            if action not in tagset:
                raise Exception()
            #check for action validity
            if not state.is_action_valid(s, action):
                print("entered not valid chunk because of",action)
                #print(pred.data.numpy())
                #print(np.sort(pred.data.numpy()))

                sorted_probs = np.sort(pred.data.numpy())[::-1]
                for i in range(1,len(sorted_probs)): # 1 start bc we already checked argmax before
                    action = "SHIFT"
                    print("try with:",tag_set_idx2name[i])
                    if state.is_action_valid(s, tag_set_idx2name[i]):
                        action =  tag_set_idx2name[i]
                        print("success with:",action)
                        break
            a = action.split("_")
            
            if  len(a) > 1:
                if a[1] == "L":
                    state.left_arc(s, action)
                elif a[1] == "R":   
                    state.right_arc(s, action)
            else:
                state.shift(s)

main()