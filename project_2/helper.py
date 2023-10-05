from scripts import state
import numpy as np
import torch

def prepare_vectors_for_training(raw_data, tagset, c_window, glove, torch_emb, pos_set_name2idx, tag_set_name2idx):
    train_mean = []
    train_concat = []
    labels = []
    for row in raw_data:
        s = state.ParseState([],row[0],[])
        for action in row[1]:
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
            w_stack = state.pad(w_stack, c_window, "token")
            p_stack = [p.pos for p in s.stack]
            p_stack = state.pad(p_stack, c_window, "postag")

            w_buffer = [w.word for w in s.parse_buffer]
            w_buffer = state.pad(w_buffer, c_window, "token")
            p_buffer = [p.pos for p in s.parse_buffer]
            p_buffer = state.pad(p_buffer, c_window, "postag")

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

            # put vecs together
            labels.append(tag_set_name2idx[action])
            train_mean.append(torch.add(w_emb_mean, p_emb_mean))
            train_concat.append(torch.add(w_emb_concat, p_emb_concat))
    return train_mean, train_concat, labels

def array_len(a):
    counter_y = 0
    for y in range(len(a)):
        counter_y += 1
        counter_x = 0
        for x in range(len(a[0])):
            counter_x += 1
    return (y, x)
    

def parse_n_predict(hidden_data, tagset, c_window, glove, torch_emb, pos_set_name2idx,
                    model, tag_set_idx2name, type):
    predictions = []
    print("size inside parse_n_predict", array_len(hidden_data))
    for row in hidden_data:
        print("size inside for", array_len(hidden_data))
        s = state.ParseState([],row,[])
        deps_predicted = []
        while not state.is_final_state(s, c_window):
            print("size inside while", array_len(hidden_data))
            w_stack = [w.word for w in s.stack]
            w_stack = state.pad(w_stack, c_window, "token")
            p_stack = [p.pos for p in s.stack]
            p_stack = state.pad(p_stack, c_window, "postag")

            w_buffer = [w.word for w in s.parse_buffer]
            w_buffer = state.pad(w_buffer, c_window, "token")
            p_buffer = [p.pos for p in s.parse_buffer]
            p_buffer = state.pad(p_buffer, c_window, "postag")

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
            pred_text = None
            if type == "mean":
                pred = model(torch.add(w_emb_mean, p_emb_mean))
                pred_text = tag_set_idx2name[round(np.argmax(pred.data.numpy()))]
            elif type == "concat":
                pred = model(torch.add(w_emb_concat, p_emb_concat))
                pred_text = tag_set_idx2name[round(np.argmax(pred.data.numpy()))]

            action = pred_text

            if action not in tagset:
                raise Exception()
            #check for action validity
            if not state.is_action_valid(s, action):
                sorted_probs = np.sort(pred.data.numpy())[::-1]
                for i in range(1,len(sorted_probs)): # 1 start bc we already checked argmax before
                    if state.is_action_valid(s, tag_set_idx2name[i]):
                        action =  tag_set_idx2name[i]
                        break
                   
            if not state.is_action_valid(s, action):
                break
            a = action.split("_")
            print("before parse\n", hidden_data)
            if  len(a) > 1:
                if a[1] == "L":
                    state.left_arc(s, action)
                elif a[1] == "R":   
                    state.right_arc(s, action)
                deps_predicted.append(action)
            else:
                state.shift(s)
                deps_predicted.append(action)
            print("after parse\n", hidden_data)
        predictions.append([deps_predicted,s.dependencies])
    return predictions 

