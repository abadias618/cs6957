from scripts import state_extra
import numpy as np
import torch

def prepare_vecs_for_extra_cred(raw_data, tagset, c_window, glove, torch_emb, pos_set_name2idx, tag_set_name2idx):
    train = []
    labels = []
    for row in raw_data:
        s = state_extra.ParseState([],row[0],[])
        for idx, action in enumerate(row[1]):
            if action not in tagset:
                raise Exception()
            # check for action validity among all tagset
            if not state_extra.is_action_valid(s, action):
                for act in tagset:
                    if state_extra.is_action_valid(s, act):
                        action = act
                        break
            # If after all there's nothing still, break        
            if not state_extra.is_action_valid(s, action):
                print("emergency break bad training file, an action is not valid")
                break
            a = action.split("_")

            if  len(a) > 1:
                if a[1] == "L":
                    state_extra.left_arc(s, action)
                elif a[1] == "R":   
                    state_extra.right_arc(s, action)
            else:
                state_extra.shift(s)

            

            #pad
            w_stack = [w.word for w in s.stack]
            w_stack = state_extra.pad(w_stack, c_window, "token")
            p_stack = [p.pos for p in s.stack]
            p_stack = state_extra.pad(p_stack, c_window, "postag")

            w_buffer = [w.word for w in s.parse_buffer]
            w_buffer = state_extra.pad(w_buffer, c_window, "token")
            p_buffer = [p.pos for p in s.parse_buffer]
            p_buffer = state_extra.pad(p_buffer, c_window, "postag")
            # get right most and left most from current 2 elements of stack
            # check that there is at least 2 elements in the stack
            # otherwise set L vector to defaults
            if len(s.stack) < 2:
                print()
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
            train.append(torch.add(w_emb_mean, p_emb_mean))

    return train, labels

def parse_n_predict(hidden_data, tagset, c_window, glove, torch_emb, pos_set_name2idx,
                    model, tag_set_idx2name, type):
    predictions = []
    for row in hidden_data:
        s = state_extra.ParseState([],row,[])
        deps_predicted = []
        while not state_extra.is_final_state(s, c_window):
            w_stack = [w.word for w in s.stack]
            w_stack = state_extra.pad(w_stack, c_window, "token")
            p_stack = [p.pos for p in s.stack]
            p_stack = state_extra.pad(p_stack, c_window, "postag")

            w_buffer = [w.word for w in s.parse_buffer]
            w_buffer = state_extra.pad(w_buffer, c_window, "token")
            p_buffer = [p.pos for p in s.parse_buffer]
            p_buffer = state_extra.pad(p_buffer, c_window, "postag")

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
            # check for action validity among probabilities
            if not state_extra.is_action_valid(s, action):
                sorted_probs = np.sort(pred.data.numpy())[::-1]
                for i in range(1,len(sorted_probs)): # 1 start bc we already checked argmax before
                    if state_extra.is_action_valid(s, tag_set_idx2name[i]):
                        action =  tag_set_idx2name[i]
                        break
            # check for action validity among all tagset
            if not state_extra.is_action_valid(s, action):
                for act in tagset:
                    if state_extra.is_action_valid(s, act):
                        action = act
                        break
            # If after all there's nothing still, break        
            if not state_extra.is_action_valid(s, action):
                break
            a = action.split("_")
            if  len(a) > 1:
                if a[1] == "L":
                    state_extra.left_arc(s, action)
                elif a[1] == "R":   
                    state_extra.right_arc(s, action)
                deps_predicted.append(action)
            else:
                state_extra.shift(s)
                deps_predicted.append(action)
        predictions.append([deps_predicted,s.dependencies])
    return predictions 