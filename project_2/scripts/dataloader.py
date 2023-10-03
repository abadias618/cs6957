def load_data(filename):
    """
    RETURNS list of lists with
    [input_tokens, pos_tags, gold_actions]
    """
    data = []
    with open(filename,"r") as file:
        for line in file:

            line = line.strip().split(sep="|||")
            input_tokens= line[0].strip().split()
            pos_tags = line[1].strip().split()
            gold_actions = line[2].strip().split()
            data.append([input_tokens, pos_tags, gold_actions])
    return data

def load_hidden(filename):
    """
    RETURNS list of lists with
    [input_tokens, pos_tags]
    """
    data = []
    with open(filename,"r") as file:
        for line in file:

            line = line.strip().split(sep="|||")
            input_tokens= line[0].strip().split()
            pos_tags = line[1].strip().split()
            data.append([input_tokens, pos_tags])
    return data

def load_tagset(filename):
    """
    RETURNS list with tagset
    """
    data = []
    with open(filename,"r") as file:
        for line in file:
            data.append(line.strip())
    return data

def load_pos_set(filename):
    """
    RETURNS list with postag set including NULL
    """
    data = []
    with open(filename,"r") as file:
        for line in file:
            data.append(line.strip())
    return data

