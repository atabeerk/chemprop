import os

from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd

def make_array(text):
    return np.fromstring(text.replace("[", "").replace("]", ""), sep=' ')


def read_descriptors(path):
    df = pd.read_csv(path)
    df['descriptors'] = df['descriptors'].apply(make_array)
    return df


def read_csvs(folder="multi_task_features"):
    df = pd.DataFrame(columns=["smiles", "descriptors"])
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.csv'):
            full_path = os.path.join(folder, filename)
            df = df.append(read_descriptors(full_path), ignore_index=True)
            print(filename, "finished")
    return df

if __name__ == "__main__":
    df = read_csvs(folder="multi_task_features_dmpnn_25_zinc_flagments")
    descriptors = df['descriptors'].tolist()
    smiles = df['smiles'].tolist()
    # create a dictionary
    # why is the dictionary length different than smiles and descriptors?
    #kmeans = MiniBatchKMeans(n_clusters=400, batch_size=400, verbose=2, ).fit(descriptors)
    #print(kmeans.cluster_centers_)