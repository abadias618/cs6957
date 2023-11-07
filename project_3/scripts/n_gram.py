def simple_ngram_probs(processed_seqs, vocab_len):
    """
    n: int. n-gram length
    chars: List[List]. List of Lists containing chars. 
    """
    stream = []
    for ngram in processed_seqs:
        stream += ngram
    stream = ",".join(str(i) for i in stream)
    
    probabilities = []
    for ngram in processed_seqs:
        s_ngram = ",".join(str(i) for i in ngram)
        num = stream.count(s_ngram)
        den = stream.count(",".join(str(i) for i in ngram[:len(ngram)-1]))
        probabilities.append((num + 1)/(den + vocab_len))
    return probabilities


def make_ngram_seq(n, seq, pad_ix):
    padding = [pad_ix] * (n-1)
    r = []
    seq = padding + seq
    for i in range(len(seq)-(n-1)):
        r.append(seq[i:i+n])
           
    return r