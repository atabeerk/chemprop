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


def read_csvs(folder):
    df = pd.DataFrame(columns=["id", "smiles", "descriptors", "cluster"])
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.csv'):
            print(filename)
            full_path = os.path.join(folder, filename)
            print(full_path)
            df = df.append(read_descriptors(full_path), ignore_index=True)
            print(filename, "finished")
    return df

df = read_csvs("/Volumes/ATABERK64")