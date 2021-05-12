import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from kmeans_clustering import read_csvs
from sklearn.cluster import MiniBatchKMeans, AffinityPropagation
from sklearn.metrics import rand_score

def murcko_clustering(folder):
    df = read_csvs(folder="../data")
    murcko_clusters = {}
    for index, row in df.iterrows():
        sm = row["smiles"]
        fw = MurckoScaffold.MurckoScaffoldSmiles(sm)
        try:
            murcko_clusters[fw].append(sm)
        except:
            murcko_clusters[fw] = [sm]
        if index % 10000 == 0:
            print(index)


def save_murcko_result(murcko_clusters):
    df_dict = {}
    for index, key in enumerate(murcko_clusters.keys()):
        if len(murcko_clusters[key]) > 400:
            for sm in murcko_clusters[key]:
                df_dict[sm] = index

if __name__ == "__main__":

    # read the compounds from murcko_labels.csv
    murcko = pd.read_csv("../data/labels/murcko_labels.csv")

    # get descriptor of each compound in murcko_labels.csv from ../data
    desc_df = read_csvs("../data")
    desc_dict = dict(zip(desc_df.smiles, desc_df.descriptors))
    chemprop_desc = {}
    for index, sm in enumerate(murcko.smiles):
        if not index % 10000:
            print(index)
        desc = desc_dict[sm]
        chemprop_desc[sm] = desc

    # cluster these compounds with kmeans clustering
    print("starting clustering")
    #kmeans = MiniBatchKMeans(n_clusters=21, verbose=1).fit(list(chemprop_desc.values()))
    ap = AffinityPropagation(verbose=True).fit(list(chemprop_desc.values()))
    # evaluate the clustering by external indices
    print("starting rand index computations")
    rand = rand_score(list(murcko.label), kmeans.labels_)