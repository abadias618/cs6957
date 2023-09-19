import sys
sys.path.insert(0, './scripts/')
from utils import *
from cbow import *


vocab = get_word2ix("./vocab.txt")
inverse_vocab = {v: k for k, v in vocab.items()}

print("\nexample vocab",{k: vocab[k] for k in list(vocab)[:5]},"\n")
files = get_files("./data/train")
print("files in dir",files[0],"\n")
data = process_data(files[0:5], 2, vocab)
print("data:",len(data[0]),len(data[1]),len(data[2]),len(data[3]),len(data[4]))

vocab_size = len(list(vocab.keys()))
embedding_dim = len(data[0])
print(f"vocab_size: {vocab_size}, embedding_size: {embedding_dim}")


model = CBOW(vocab_size, embedding_dim)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print(make_sentence_vector(['stormy','nights','when','the'], vocab))

#train
for epoch in range(1):
    epoch_loss = 0
    for sentence, target in data:
        model.zero_grad()
        sentence_vector = make_sentence_vector(sentence, vocab)  
        log_probs = model(sentence_vector)
        loss = loss_function(log_probs, torch.tensor(
        [vocab[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.data
    print('Epoch: '+str(epoch)+', Loss: ' + str(epoch_loss.item()))