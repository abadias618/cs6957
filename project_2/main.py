from scripts import dataloader
from scripts import state
from scripts import evaluate
from helper import *
from Mean import *
from Concat import *
import numpy as np
import random
from copy import deepcopy
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
    pos_set, pos_set_idx2name, pos_set_name2idx = dataloader.load_pos_set("./data/pos_set.txt")
    tagset, tag_set_idx2name, tag_set_name2idx = dataloader.load_tagset("./data/tagset.txt")

    C_WINDOW = 2
    NUMBER_OF_POSTAGS = len(pos_set)
    NUMBER_OF_ACTIONS = len(tagset)

    data = [] #tokens-dependencies-ParseState
    #put data into objs
    for row in complete_data[:]:
        tokens = \
        [state.Token(i+1,input_token,pos_tag) for i, (input_token, pos_tag) in enumerate(zip(row[0], row[1]))]
        data.append([tokens, row[2]])

    
    GLOVE_CONFIG=[("6B",50),("6B", 300),("42B",300),("840B", 300)]

    overall_score = float('-inf')
    best_model = None
    for gc in GLOVE_CONFIG[0]:
        name = gc[0]
        dim = gc[1]
        
        # Glove embeddings
        glove = torchtext.vocab.GloVe(name=name, dim=dim)
        # Torch embeddings
        torch_emb = nn.Embedding(NUMBER_OF_POSTAGS, dim)
        
        train_mean, train_concat, labels = prepare_vectors_for_training(deepcopy(data), tagset=tagset,
                                                        c_window=C_WINDOW, glove=glove,
                                                        torch_emb=torch_emb,
                                                        pos_set_name2idx=pos_set_name2idx,
                                                        tag_set_name2idx=tag_set_name2idx)
        #READY DATA
        dataset_mean = TensorDataset(torch.stack(train_mean), torch.tensor(labels))
        dataloader_mean = DataLoader(dataset_mean, batch_size = 1024, shuffle=False)

        dataset_concat = TensorDataset(torch.stack(train_concat), torch.tensor(labels))
        dataloader_concat = DataLoader(dataset_concat, batch_size = 1024*4, shuffle=False)

        test_data = dataloader.load_data("./data/test.txt")
        obj_test_data = [] #tokens-dependencies-ParseState
        gold_actions = []
        word_lists = []
        #put data into objs
        for row in test_data[:]:
            tokens = \
            [state.Token(i+1,input_token,pos_tag) for i, (input_token, pos_tag) in enumerate(zip(row[0], row[1]))]
            obj_test_data.append(tokens)
            word_lists.append(row[0])
            gold_actions.append(row[2])
        
        print("FINALIZED DATA CREATION\n\n")


        
        m_or_c = None
        for lr in [0.01, 0.001, 0.0001][2]:

            

            # Create Models
            model_mean = Mean(dim, NUMBER_OF_ACTIONS)
            #print("MODEL_MEAN CREATED\n",model_mean)
            model_concat = Concat(dim, NUMBER_OF_ACTIONS)
            #print("MODEL_CONCAT CREATED\n",model_concat)
            loss_function = nn.CrossEntropyLoss()
            optimizer_mean = torch.optim.Adam(model_mean.parameters(), lr=lr)
            optimizer_concat = torch.optim.Adam(model_concat.parameters(), lr=lr)

            # train
            #model_mean = train_mean_model(dataloader_mean, model_mean, loss_function, optimizer_mean, epochs=1)
            model_concat = train_concat_model(dataloader_concat, model_concat, loss_function, optimizer_concat, epochs=1)
            
            # UAS - LAS
            #m_predictions_test = parse_n_predict(hidden_data=deepcopy(obj_test_data), tagset=tagset,
            #                            c_window=C_WINDOW, glove=glove,
            #                            torch_emb=torch_emb,
            #                            pos_set_name2idx=pos_set_name2idx, model=model_mean,
            #                            tag_set_idx2name=tag_set_idx2name, type="mean")
            c_predictions_test = parse_n_predict(hidden_data=deepcopy(obj_test_data), tagset=tagset,
                                        c_window=C_WINDOW, glove=glove,
                                        torch_emb=torch_emb,
                                        pos_set_name2idx=pos_set_name2idx, model=model_concat,
                                        tag_set_idx2name=tag_set_idx2name, type="concat")

            #m_uas_las = evaluate.compute_metrics(word_lists, gold_actions, 
            #                                [p[0] for p in m_predictions_test], C_WINDOW)
            
            c_uas_las = evaluate.compute_metrics(word_lists, gold_actions, 
                                            [p[0] for p in c_predictions_test], C_WINDOW)
            

            #print("mean model UAS-LAS", m_uas_las)
            print("concat model UAS-LAS", c_uas_las)
            print("current lr", lr)
            #f m_uas_las[1] > c_uas_las[1]:
            #   if m_uas_las[1] > overall_score:
            #       overall_score = m_uas_las[1]
            #       best_model = model_mean
            #       m_or_c = "mean"
            #   elif c_uas_las[1] > overall_score:
            #       overall_score = c_uas_las[1]
            #       best_model = model_concat
            #       m_or_c = "concat"
            best_model = model_concat       

    model = best_model


    hidden_data = dataloader.load_hidden("./data/hidden.txt")
    obj_hidden_data = [] #tokens-dependencies-ParseState
    #put data into objs
    for row in hidden_data[:]:
        tokens = \
        [state.Token(i+1,input_token,pos_tag) for i, (input_token, pos_tag) in enumerate(zip(row[0], row[1]))]
        obj_hidden_data.append(tokens)
    predictions_hidden = parse_n_predict(obj_hidden_data, tagset=tagset,
                                 c_window=C_WINDOW, glove=glove,
                                 torch_emb=torch_emb,
                                 pos_set_name2idx=pos_set_name2idx, model=model,
                                 tag_set_idx2name=tag_set_idx2name,type=m_or_c)
    print("predictions for hidden finished")
    # create .txt file
    with open("results.txt","w") as file:
        for p in [x[0] for x in predictions_hidden]:
            file.write(" ".join(p) + "\n")

    # get q4 dependency trees
    q4_data = dataloader.load_hidden("./data/q4.txt")
    obj_q4_data = [] #tokens-dependencies-ParseState
    #put data into objs
    for row in q4_data:
        tokens = \
        [state.Token(i+1,input_token,pos_tag) for i, (input_token, pos_tag) in enumerate(zip(row[0], row[1]))]
        obj_q4_data.append(tokens)

    predictions_q4 = parse_n_predict(obj_q4_data, tagset=tagset,
                                 c_window=C_WINDOW, glove=glove,
                                 torch_emb=torch_emb,
                                 pos_set_name2idx=pos_set_name2idx, model=model,
                                 tag_set_idx2name=tag_set_idx2name, type=m_or_c)
    print("ANSWER for Q4")
    deps = [d[1] for d in predictions_q4]
    for d, words in zip(deps, q4_data):
        print(f"For {words[0]} the dependencies are:")
        print([str(x.source.word+"-"+x.source.pos+" >> "+x.label+" >> "+x.target.word+"-"+x.target.pos) for x in d])
        print()
    print("MAIN FINISHED")
main()