import os
import sys
sys.path.insert(0, './scripts/')
from utils import *
import numpy as np
from numpy.linalg import norm

#load vecs to memory
vocab = get_word2ix("./vocab.txt")
with open("./embeddings.txt","r") as file:
    data = []
    for i, line in enumerate(file.readlines()):
        if i == 0:
            continue
        l = line.strip().split()[1:]
        data.append([float(n) for n in l])

data = np.array(data)
ex_a = [[("cat","tiger"),("plane","human")],
        [("my","mine"),("happy", "human")],
        [("happy","cat"),("king", "princess")],
        [("ball","racket"),("good", "ugly")],
        [("cat","racket"),("good", "bad")],
    ]
for t in ex_a:
    A_1 = data[vocab[t[0][0]]]
    B_1 = data[vocab[t[0][1]]]
    cosine_1 = np.dot(A_1,B_1)/(norm(A_1)*norm(B_1))
    A_2 = data[vocab[t[1][0]]]
    B_2 = data[vocab[t[1][1]]]
    cosine_2 = np.dot(A_2,B_2)/(norm(A_2)*norm(B_2))
    print(f"{t[0][0]},{t[0][1]} and {t[1][0]},{t[1][1]}")
    if cosine_1 > cosine_2:
        print("left pair")
    elif cosine_2 > cosine_1:
        print("right pair")
    else:
        print("equal")


ex_b = [[("king","queen"),"man"],
        [("king","queen"),"prince"],
        [("king","man"),"queen"],
        [("woman","man"),"princess"],
        [("prince","princess"),"man"],
    ]

for t in ex_b:
    w_a = data[vocab[t[0][0]]]
    w_b = data[vocab[t[0][1]]]
    w_c = data[vocab[t[1]]]
    w = (w_b - w_a) + w_c
    most_sim = float("-inf")
    res = None
    for row in data:
        sim = np.dot(w,row)/(norm(w)*norm(row))
        if sim > most_sim:
            most_sim = sim
            res = np.where(data == row)[0][0]
    print(f"{t[1]}",res,list(vocab.keys())[res])

#ex_3a
ex_3a = [[("early","late"),("secrets","chase")],
         [("building","build"),("destroy","truly")] 
        ]
for t in ex_3a:
    A_1 = data[vocab[t[0][0]]]
    B_1 = data[vocab[t[0][1]]]
    cosine_1 = np.dot(A_1,B_1)/(norm(A_1)*norm(B_1))
    A_2 = data[vocab[t[1][0]]]
    B_2 = data[vocab[t[1][1]]]
    cosine_2 = np.dot(A_2,B_2)/(norm(A_2)*norm(B_2))
    print(f"{t[0][0]},{t[0][1]} and {t[1][0]},{t[1][1]}")
    if cosine_1 > cosine_2:
        print("left pair")
    elif cosine_2 > cosine_1:
        print("right pair")
    else:
        print("equal")
ex_3b = [[("tree","plant"),"flower"],
        [("good","bad"),"white"],
    ]

for t in ex_3b:
    w_a = data[vocab[t[0][0]]]
    w_b = data[vocab[t[0][1]]]
    w_c = data[vocab[t[1]]]
    w = (w_b - w_a) + w_c
    most_sim = float("-inf")
    res = None
    for row in data:
        sim = np.dot(w,row)/(norm(w)*norm(row))
        if sim > most_sim:
            most_sim = sim
            res = np.where(data == row)[0][0]
    print(f"{t[1]}",res,list(vocab.keys())[res])

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
extra = ["horse", "cat", "dog", "i", "he", "she", "it", "her", "his", "our", "we", "in", "on",
"from", "to", "at", "by", "man", "woman", "boy", "girl", "king", "queen", "prince","princess"]
df = []
for w in extra:
    df.append(data[vocab[w]])

pca = PCA(n_components=2)
X = pca.fit(df).transform(df)
#3plt.scatter([x[0] for x in df], [x[1] for x in df])
#print(X)
#plt.savefig('2D.png')
        
z = [x[0] for x in df]
y = [x[1] for x in df]
fig, ax = plt.subplots()
ax.scatter(z, y)

for i, txt in enumerate(extra):
    ax.annotate(txt, (z[i], y[i]))
        
plt.savefig('2D.png')