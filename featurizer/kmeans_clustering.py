import os, random
from collections import Counter

from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd

def make_array(text):
    return np.fromstring(text.replace("[", "").replace("]", ""), sep=' ')


def read_one_csv(path):
    df = pd.read_csv(path)
    df['descriptors'] = df['descriptors'].apply(make_array)
    return df


def read_csvs(folder):
    df = pd.DataFrame(columns=["smiles", "descriptors"])
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.csv'):
            full_path = os.path.join(folder, filename)
            df = df.append(read_one_csv(full_path), ignore_index=True)
            print(filename, "finished")
    return df

def kmeans_clustering(path, save_dir):
    df = read_csvs(path)
    print(len(df))
    # create a dictionary
    smiles_descriptors_dict = dict(zip(df.smiles, df.descriptors))
    print("Total number of molecules:", len(smiles_descriptors_dict))
    # why is the length of this dictionary different than smiles and descriptors? Because of DUPLICATES!
    SUBSET_SIZE = 50000000
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
        kmeans = MiniBatchKMeans(n_clusters=39, batch_size=800, verbose=1, init=cs).fit(list(dict_subset.values()))
        cs = kmeans.cluster_centers_
        # create a new df with row1:smiles, row2:cluster label
        smiles_label_df = pd.DataFrame(list(zip(dict_subset.keys(), kmeans.labels_)), columns=["smiles", "labels"])
        # save this df as csv
        if save_dir:
            smiles_label_df.to_csv("../data/labels/" + save_dir + ".csv")

        return kmeans

if __name__ == "__main__":

    folder = "../data/zinc15-minor-targets-features"
    true_labels = {}
    for index, file in enumerate(sorted(os.listdir(folder))):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            for _, row in df.iterrows():
                true_labels[row["smiles"]] = index

    df = pd.read_csv("../data/labels/zinc15-minor-targets.csv")
    ordered_true_labels = {}
    for _, row in df.iterrows():
        ordered_true_labels[row["smiles"]] = true_labels[row["smiles"]]

    df = pd.read_csv("../data/labels/zinc15-minor-targets.csv", index_col=0)
    fps = {}
    for index, row in df.iterrows():
        if not (index % 10000):
            print(index)
        m = Chem.MolFromSmiles(row["smiles"])
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024).ToBitString()
        fps[row["smiles"]] = np.frombuffer(fp.encode(), dtype='u1') - ord('0')

    print("fps completed...")
    indices = random.sample(range(0, len(fps)), 50000) # randomly selected indices
    selected_fps = [list(fps.values())[i] for i in indices]
    print("values completed")
    selected_smiles = [list(fps.keys())[i] for i in indices]
    print("keys completed")
    selected_true_labels = []
    for sm in selected_smiles:
        selected_true_labels.append(true_labels[sm])

    print("starting pca")
    pca = PCA(n_components=20).fit_transform(selected_fps)
    print("pca completed...")

    sil_scores = []
    rand_scores = []
    sil_scores_120 = []
    rand_scores_120 = []
    less_than_120_k = []
    for i in range(20, 80, 5):
        # need to sample by index
        print("ITERATION:", i)
        kmeans = MiniBatchKMeans(n_clusters=i, batch_size=800, init="k-means++").fit(pca)
        print("kmeans completed...")
        rs = adjusted_rand_score(selected_true_labels, kmeans.labels_)
        print("adjusted rand score:", rs)
        rand_scores.append(rs)
        ss = silhouette_score(pca, kmeans.labels_)
        print("silhouette score:", ss)
        sil_scores.append(ss)
        counter = Counter(kmeans.labels_)
        less_than_120 = []
        for k in range(len(pca)):
            if counter[kmeans.labels_[k]] < 120:
                continue
            less_than_120.append(pca[k])
        print("len_120:", len(less_than_120))
        if len(less_than_120) < len(pca):
            less_than_120_k.append(i)
        kmeans = MiniBatchKMeans(n_clusters=i, batch_size=800, init="k-means++").fit(less_than_120)
        print("kmeans completed...")
        #rs = adjusted_rand_score(selected_true_labels, kmeans.labels_)
        #print("adjusted rand score:", rs)
        #rand_scores_120.append(rs)
        ss = silhouette_score(less_than_120, kmeans.labels_)
        print("silhouette score:", ss)
        sil_scores_120.append(ss)







