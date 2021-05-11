import ast
import pickle
import pandas as pd
import os


def make_array(text):
    return pickle.loads(ast.literal_eval(text))


def read_descriptors(path):
    df = pd.read_csv(path, names=["id", "smiles", "descriptors", "cluster"])
    df['descriptors'] = df['descriptors'].apply(make_array)
    return df


def read_csv(path):
    df = pd.DataFrame(columns=["id", "smiles", "descriptors", "cluster"])
    df = read_descriptors(path)
    return df