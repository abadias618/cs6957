from scripts import state
import torch

def prepare_vectors_for_training(raw_data, tagset, C_WINDOW, glove, torch_emb, pos_set_name2idx, tag_set_name2idx):
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

            # put vecs together
            labels.append(tag_set_name2idx[action])
            train_mean.append(torch.add(w_emb_mean, p_emb_mean))
            train_concat.append(torch.add(w_emb_concat, p_emb_concat))
    return train_mean, train_concat, labels

def get_vectors():
    return