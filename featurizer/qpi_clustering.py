import random, collections

import pandas as pd
from sklearn.cluster import MiniBatchKMeans

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

if __name__ == "__main__":
    true_labels = get_true_labels("../data/chembl27-with-decoys/smiles/CHEMBL1862_actives_decoys.csv")
    df = read_one_csv("../data/chembl27-with-decoys/features/CHEMBL1862_actives_decoys.csv")
    descriptors = dict(zip(df["smiles"], df["descriptors"]))
    print(list(descriptors.keys())[3])
    descriptors = shuffle_dict(descriptors)
    smiles = list(descriptors.keys()) # smiles corresponding to X
    X = list(descriptors.values())
    kmeans_labels = MiniBatchKMeans(n_clusters=2, batch_size=800, init="k-means++").fit(X).labels_
    clusters_2d = get_clusters_2d(smiles, kmeans_labels)
    qpi, _ = calculate_QPI(true_labels, clusters_2d)
    print(qpi)
    print(collections.Counter(true_labels.values()))


