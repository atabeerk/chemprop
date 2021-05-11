from rdkit.Chem.Scaffolds import MurckoScaffold
from kmeans_clustering import read_csvs
#MurckoScaffold.MurckoScaffoldSmiles(sm)

if __name__ == "__main__":
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

        df_dict = {}
        for index, key in enumerate(murcko_clusters.keys()):
            if len(murcko_clusters[key]) > 400:
                for sm in murcko_clusters[key]:
                    df_dict[sm] = index

        # read the compounds from murcko_labels.csv
        # get descriptor of each compound in murcko_labels.csv from ../data
        # cluster these compounds with kmeans clustering
        # evaluate the clustering by external indices