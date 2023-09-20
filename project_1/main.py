import os
import sys
sys.path.insert(0, './scripts/')
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from utils import *
from cbow import *


vocab = get_word2ix("./vocab.txt")
print("vocab length:",len(vocab.keys()))
WINDOW = 5
#print("\nexample vocab",{k: vocab[k] for k in list(vocab)[:5]},"\n")
files = get_files("./data/train")
#print("files in dir",files[0],"\n")
vectors = process_data(files[0:2], WINDOW, vocab)
print("vectors length:",len(vectors))
#print("\ndata example lengths:",len(vectors[0]),len(vectors[1]))#,len(vectors[2]))
def vec2data(vecs, win):
    df = []
    labels = []
    for v in vecs:
        for i in range(win, len(v) - win):
            words = v[i - win:i] + v[i + 1:i + win + 1]
            df.append(words)
            labels.append(v[i])
    return df, labels
data, labels = vec2data(vectors, WINDOW)
#print("\ndata length=",len(data))
#print("\nexample data",data[0],"\n")
#print("\nexample data",data[1],"\n")
#print("\nexample data",data[2],"\n")

VOCAB_SIZE = len(list(vocab.keys()))
EMBEDDING_DIM = 100
print(f"VOCAB_SIZE: {VOCAB_SIZE}, embedding_size: {EMBEDDING_DIM}")






dataset = TensorDataset(torch.tensor(data), torch.tensor(labels))
dataloader = DataLoader(dataset, batch_size = 64, shuffle=False)
for lr in [0.01, 0.001, 0.0001]:
    model = CBOW(VOCAB_SIZE, EMBEDDING_DIM)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    overall_loss = float('inf')
    #train
    for epoch in range(1):
        with tqdm(dataloader) as tepoch:
            
            for sentence, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                model.zero_grad()
                sentence_vector = sentence
                log_probs = model(sentence)
                loss = loss_function(log_probs, target)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
            print('Epoch# '+str(epoch)+' - Loss: ' + str(loss.item()))
            if loss < overall_loss:
                overall_loss = loss
                if os.path.exists("./embeddings.txt"):
                    os.remove("embeddings.txt")
                with open("embeddings.txt","w") as f:
                    f.write(f"{str(VOCAB_SIZE)} {str(EMBEDDING_DIM)}\n")
                    for w in vocab.keys():
                        word = w
                        embeds = model.get_word_emdedding(vocab, word)[0].tolist()
                        embeds = " ".join(str(e) for e in embeds)
                        f.write(f"{word} {embeds}\n")


