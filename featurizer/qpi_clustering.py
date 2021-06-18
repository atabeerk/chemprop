import random, collections

import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from rdkit import Chem
from rdkit.Chem import AllChem

from kmeans_clustering import read_one_csv

def get_true_labels(smiles_file):
    df = pd.read_csv(smiles_file)
    return dict(zip(df["Smiles"], df["Label"]))


def shuffle_dict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    values = [d[k] for k in keys]

    return dict(zip(keys, values))


def get_clusters_2d(smiles, labels):
    # Assumes labels start from 0 and go to len(labels)-1
    clusters = []
    for i in range(max(labels)+1):
        clusters.append([])

    for i in range(len(labels)):
        clusters[labels[i]].append(smiles[i])

    return clusters


def calculate_QPI(true_labels, clusters_2d):
    cnt = collections.Counter(list(true_labels.values()))
    set_ratio = cnt[1] / cnt[0]
    cluster_labels = []
    correct_preds = 0
    for cluster in clusters_2d:
        tl = [true_labels[smiles] for smiles in cluster]
        cnt = collections.Counter(tl)
        # if there are no inactives, set cluster_ratio to 1 which is greater than set ratio
        cluster_ratio = cnt[1] / cnt[0] if cnt[0] else 1
        c_label = 1 if cluster_ratio > set_ratio else 0
        cluster_labels.append(c_label)
        correct_preds += cnt[1] if c_label else cnt[0]

    qpi = correct_preds / len(true_labels)
    return qpi, cluster_labels


def get_ecfp4_vectors(smiles_list):
    ecfp4_list = []
    for smiles in smiles_list:
        m = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024).ToBitString()
        ecfp4_list.append(np.frombuffer(fp.encode(), dtype='u1') - ord('0'))

    return ecfp4_list


if __name__ == "__main__":
    true_labels = get_true_labels("../data/chembl27-with-decoys/smiles/CHEMBL1862_actives_decoys.csv")
    df = read_one_csv("../data/chembl27-with-decoys/features/CHEMBL1862_actives_decoys.csv")

    # Using Chemprop descriptors
    descriptors = dict(zip(df["smiles"], df["descriptors"]))
    descriptors = shuffle_dict(descriptors)
    smiles = list(descriptors.keys()) # smiles corresponding to X
    X = list(descriptors.values())
    kmeans_labels = MiniBatchKMeans(n_clusters=200, batch_size=800, init="k-means++").fit(X).labels_
    clusters_2d = get_clusters_2d(smiles, kmeans_labels)
    qpi, _ = calculate_QPI(true_labels, clusters_2d)
    print(qpi)
    print(collections.Counter(true_labels.values()))

    # Using ECFP4 vectors
    ecfp4_list = get_ecfp4_vectors(smiles)
    ecfp4_dict = dict(zip(smiles, ecfp4_list))
    X = list(ecfp4_dict.values())
    kmeans_labels = MiniBatchKMeans(n_clusters=200, batch_size=800, init="k-means++").fit(X).labels_
    clusters_2d = get_clusters_2d(smiles, kmeans_labels)
    qpi, _ = calculate_QPI(true_labels, clusters_2d)
    print(qpi)
    print(collections.Counter(true_labels.values()))


