import os, random

import pandas as pd
from rdkit.ML.Cluster import Butina
from rdkit import Chem, DataStructs
from rdkit import DataStructs
from rdkit.Chem import AllChem


def ClusterFps(fps, cutoff=0.2):
    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    for i in range(1,nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
        dists.extend([1-x for x in sims])

    # now cluster the data:
    cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    return cs


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
    indices = random.sample(range(0, len(df)), 10000) # randomly selected indices
    fps = {}
    for index, row in df.iterrows():
        if index in indices:
            m = Chem.MolFromSmiles(row["smiles"])
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
            fps[row["smiles"]] = fp

    selected_fps = list(fps.values())
    selected_smiles = list(fps.keys())
    selected_true_labels = []
    for sm in selected_smiles:
        selected_true_labels.append(true_labels[sm])

    butina = ClusterFps(selected_fps, 0.8)