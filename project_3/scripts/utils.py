import glob

def get_files(path):
    """ Returns a list of text files in the 'path' directory.
    Input
    ------------
    path: str or pathlib.Path. Directory path to load files from. 

    Output
    -----------
    file_list: List. List of paths to text files
    """
    file_list =  list(glob.glob(f"{path}/*.txt"))
    return file_list




def convert_line2idx(line, vocab):
    """ Converts a string into a list of character indices
    Input
    ------------
    line: str. A line worth of data
    vocab: dict. A dictionary mapping characters to unique indices

    Output
    -------------
    line_data: List[int]. List of indices corresponding to the characters
                in the input line.
    """
    line_data = []
    for charac in line:
        if charac not in vocab.keys():
            line_data.append(vocab["<unk>"])
        else:
            line_data.append(vocab[charac])
    return line_data




def convert_files2idx(files, vocab):
    """ This method iterates over files. In each file, it iterates over
    every line. Every line is then split into characters and the characters are 
    converted to their respective unique indices based on the vocab mapping. All
    converted lines are added to a central list containing the mapped data.
    Input
    --------------
    files: List[str]. List of files in a particular split
    vocab: dict. A dictionary mapping characters to unique indices

    Output
    ---------------
    data: List[List[int]]. List of lists where each inner list is a list of character
            indices corresponding to a line in the training split.
    """
    data = []

    for file in files:
        with open(file) as f:
            lines = f.readlines()
        
        for line in lines:
            toks = convert_line2idx(line, vocab)
            data.append(toks)

    return data

def make_sub_seqs(n, seqs, pad_ix):
    """
    n: int. length of sequences to output
    seqs: List[List]. list of lists

    Output
    ---------------
    sequences of length n
    """
    train_seqs = []
    target_seqs = []
    for s in seqs:
        l = len(s) // n
        
        if l == 0:
            pad_len = n-len(s)
            train_seqs.append(s+([pad_ix] * pad_len))
            target_seqs.append(s[1:]+([pad_ix] * (pad_len+1)))
            continue

        start = 0
        end = n
        for _ in range(l):
            train_seqs.append(s[start:end])
            if end+1 <= len(s):
                target_seqs.append(s[start+1:end+1])
            else:
                target_seqs.append(s[start+1:end]+[pad_ix])
            start = end
            end = end + n
        # at this point start will hold the end value
        if start < len(s):
            pad_len = n-(len(s)-start)
            train_seqs.append((s[start:] + ([pad_ix] * pad_len)))

            target_seqs.append((s[start+1:] + ([pad_ix] * (pad_len+1))))


        

    return train_seqs, target_seqs
from math import log2

def perplexity(probs, t):
    summation = 0
    for p in probs:
        summation += log2(p)
    
    return 2**(-(1/t)*summation)