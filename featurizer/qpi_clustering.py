import random, collections, time, sys

import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from rdkit import Chem
from rdkit.Chem import AllChem

from kmeans_clustering import read_one_csv
from butina_clustering import ClusterFps

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

def true_QPI(true_labels, clusters_2d):
    cnt = collections.Counter(list(true_labels.values()))
    set_ratio = cnt[1] / cnt[0]
    cluster_labels = []
    correct_preds = 0
    p, q, r, s = 0, 0, 0, 0
    for cluster in clusters_2d:
        tl = [true_labels[smiles] for smiles in cluster]
        cnt = collections.Counter(tl)
        # if there are no inactives, set cluster_ratio to 1 which is greater than set ratio
        cluster_ratio = cnt[1] / cnt[0] if cnt[0] else 1
        c_label = 1 if cluster_ratio > set_ratio else 0
        cluster_labels.append(c_label)
        if c_label == 1:
            p += cnt[1]
            q += cnt[0]
        elif c_label == 0:
            r += cnt[1]
        if len(tl) == 1:
            s += cnt[1]

    qpi = p / p + q + r + s
    return qpi, cluster_labels


def get_ecfp4_vectors(smiles_list):
    ecfp4_list = []
    for smiles in smiles_list:
        m = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
            #ToBitString()
        #ecfp4_list.append(np.frombuffer(fp.encode(), dtype='u1') - ord('0'))
        ecfp4_list.append(fp)
    return ecfp4_list

def sample_n(n, true_labels, descriptors):
    if n > len(true_labels):
        raise Exception('n greater than length of true labels')

    actives = []
    inactives = []
    for smiles in true_labels.keys():
        if true_labels[smiles] == 1:
            actives.append(smiles)
        else:
            inactives.append(smiles)

    selected_actives = random.sample(actives, n)
    selected_inactives = random.sample(inactives, n*60)
    all_selected = selected_actives + selected_inactives

    desc = {}
    labels = {}
    for smiles in all_selected:
        desc[smiles] = descriptors[smiles]
        labels[smiles] = true_labels[smiles]

    return labels, desc


def fix_butina_output(butina_output, smiles):
    # smiles ordering should be the same as indices of butina output
    clusters_2d, labels_1d = [], np.zeros(len(smiles))
    for cluster in range(len(butina_output)):
        tmp = []
        for index in butina_output[cluster]:
            tmp.append(smiles[index])
            labels_1d[index] = cluster
        clusters_2d.append(tmp)

    return clusters_2d, labels_1d


if __name__ == "__main__":
    active_df = read_one_csv("../data/DUD-E/chemprop_features/reni_actives.csv") # 1947
    decoy_df = read_one_csv("../data/DUD-E/chemprop_features/reni_decoys.csv")
    df = pd.concat([active_df, decoy_df])
    labels = ([1] * len(active_df)) + ([0] * len(decoy_df))
    true_labels = dict(zip(df["smiles"], labels))

    # Using Chemprop descriptors
    chemprop_qpis, chemprop_ss = [], []
    descriptors = dict(zip(df["smiles"], df["descriptors"]))
    # sample desired number of points
    # sampled_labels, sampled_descriptors = sample_n(333, true_labels, descriptors)
    descriptors = shuffle_dict(descriptors)
    smiles = list(descriptors.keys()) # smiles corresponding to X
    X = list(descriptors.values())
    start = time.time()
    for k in range(10, 1010, 10):
        clustering_labels = MiniBatchKMeans(n_clusters=k, batch_size=800, init="k-means++").fit(X).labels_
        clusters_2d = get_clusters_2d(smiles, clustering_labels)
        qpi, _ = true_QPI(true_labels, clusters_2d)
        print(k, qpi)
        chemprop_qpis.append(qpi)
        # silhouette, rand
        chemprop_ss.append(silhouette_score(X, clustering_labels, sample_size=20000))
    chemprop_time = time.time() - start
    print(chemprop_time)

    # Using ECFP4 vectors
    ecfp4_qpis, ecfp4_ss = [], []
    ecfp4_list = get_ecfp4_vectors(smiles)
    ecfp4_dict = dict(zip(smiles, ecfp4_list))
    X = list(ecfp4_dict.values())
    start = time.time()
    for k in range(10, 1010, 10):
        clustering_labels = MiniBatchKMeans(n_clusters=k, batch_size=800, init="k-means++").fit(X).labels_
        clusters_2d = get_clusters_2d(smiles, clustering_labels)
        qpi, _ = true_QPI(true_labels, clusters_2d)
        print(k, qpi)
        ecfp4_qpis.append(qpi)
        # silhouette, rand
        ecfp4_ss.append(silhouette_score(X, clustering_labels, sample_size=20000))
    ecfp4_time = time.time() - start
    print("chemprop ss1 =", chemprop_ss)
    print("chemprop qpi1 =", chemprop_qpis)
    print("ecfp4 ss2 =", ecfp4_ss)
    print("ecfp4 qpi2 =", ecfp4_qpis)
    print("chemprop_time:", chemprop_time)
    print("ecfp4_time:", ecfp4_time)


