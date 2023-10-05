import torch
from torch import nn
from tqdm import tqdm

class Concat(torch.nn.Module):
    def __init__(self, embedding_dim, actions_dim):
        super(Concat, self).__init__()
        
        self.linear1 = nn.Linear(embedding_dim, 200)
        self.linear2 = nn.Linear(200, actions_dim)

        self.activation_function1 = nn.ReLU()
        self.activation_function2 = nn.Softmax(dim=-1)

    def forward(self, inputs):
        out = self.activation_function1(inputs)
        print("concat inputs",inputs.size())
        out = self.linear1(out)
        out = self.linear2(out)    
        out = self.activation_function2(out)
        return out
    
def train_concat_model(dataloader, model, loss_f, optimizer, epochs):
    # Train
    for epoch in range(epochs):
        with tqdm(dataloader) as tepoch:
            for vector, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                model.zero_grad()
                log_probs = model(vector)
                loss = loss_f(log_probs, target)
                loss.backward(retain_graph=True)
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
            print('Epoch# '+str(epoch)+' - Loss: ' + str(loss.item()))   
    print("FINALIZED training\n\n")
    return model
