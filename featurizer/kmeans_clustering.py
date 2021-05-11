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


def read_csvs(folder):
    df = pd.DataFrame(columns=["smiles", "descriptors"])
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.csv'):
            full_path = os.path.join(folder, filename)
            df = df.append(read_descriptors(full_path), ignore_index=True)
            print(filename, "finished")
    return df

if __name__ == "__main__":
    # Should I read 100 files, minibatch cluster and read another 100...?
    df = read_csvs(folder="../data")
    # create a dictionary
    smiles_descriptors_dict = dict(zip(df.smiles, df.descriptors))
    print("Total number of molecules:", len(smiles_descriptors_dict))
    # why is the length of this dictionary different than smiles and descriptors? Because of DUPLICATES!
    SUBSET_SIZE = 10000000
    start = 0
    cs = "k-means++"
    for i in range(SUBSET_SIZE, len(smiles_descriptors_dict)+SUBSET_SIZE, SUBSET_SIZE):
        # print step:
        print("ITERATION:", i/SUBSET_SIZE)
        # set i to len(dict) if i > len(dict)
        if i > len(smiles_descriptors_dict):
            i = len(smiles_descriptors_dict)
        # take a subset of the dictionary
        print("Taking subset of the dictionary:", start, i)
        dict_subset = {key: smiles_descriptors_dict[key] for key in list(smiles_descriptors_dict.keys())[start:i]}
        start = i
        # run minibatch kmeans
        print("starting minibatch kmeans")
        kmeans = MiniBatchKMeans(n_clusters=50, batch_size=800, verbose=1, init=cs).fit(list(dict_subset.values()))
        cs = kmeans.cluster_centers_
        # create a new df with row1:smiles, row2:cluster label
        smiles_label_df = pd.DataFrame(list(zip(dict_subset.keys(), kmeans.labels_)), columns=["smiles", "labels"])
        # save this df as csv
        smiles_label_df.to_csv("../data/chembl_labels_" + str(int(i/SUBSET_SIZE)) + ".csv")

        secilmis compoundlarin Murcko scaffoldlarini cikar
        bu scaffoldlar uzerinden cluster et,

