import csv
import random
random.seed(1419615)

def load_rte_data(filename: str) -> list:
    with open(filename, newline="") as file:
        reader = csv.reader(file, delimiter=",")
        processed = []
        for row in reader:
            processed.append(row)

    return processed[1:] # avoid returning the column names

def load_sst2_data(filename: str) -> list:
    with open(filename, newline="") as file:
        reader = csv.reader(file, delimiter=",")
        processed = []
        for row in reader:
            processed.append(row[0])

    return processed[1:] # avoid returning the column names

def get_baseline(quantity: int) -> list:
    return [random.randint(0,1) for _ in range(quantity)]