import os, zipfile, sys

import pandas as pd
import numpy as np

from chemprop.nn_utils import compute_molecule_vectors
from chemprop.utils import load_checkpoint
from chemprop.data import get_data_from_smiles


def featurize_file(file_path, output_path, pretrained_model):
    input_df = pd.read_csv(file_path, sep="\t", usecols=[0])
    smiles_list = input_df[input_df.columns[0]].tolist()
    data = get_data_from_smiles(smiles=[[smiles] for smiles in smiles_list])
    descriptors = compute_molecule_vectors(model=pretrained_model, data=data, batch_size=64)
    if not counter(file_path) == len(descriptors):
        raise Exception("Number of input smiles and output descriptors are not equal.", counter(file_path), "smiles",
                        len(descriptors), "descriptors")

    smiles_descriptors_dict = {'smiles': smiles_list, 'descriptors': descriptors}
    output_df = pd.DataFrame(smiles_descriptors_dict)
    output_df.to_csv(output_path, mode='a+', header=not os.path.exists(output_path), encoding="ascii", index=False)


def make_array(text):
    return np.fromstring(text, sep=' ')


def read_descriptors(path):
    df = pd.read_csv(path, dtype={'descriptors': np.float32})
    df['descriptors'] = df['descriptors'].apply(make_array)
    print(df)


def featurize(dataset_folder):
    counter = 0
    model = load_checkpoint("../multi_task_subfamily_dmpnn_25/fold_0/model_0/model.pt")
    files = sorted(os.listdir(dataset_folder))
    for filename in files:
        counter += 1
        print(files)
        print(filename)
        print(counter, "starting")
        if counter < 798: # Should change to 0 as I deleted the read files
            continue
        return
        output_path = "multi_task_features_dmpnn_25_zinc_flagments/file_" + str(counter // 4) + ".csv"
        if filename.endswith(".txt") and not os.stat(os.path.join("splitted",filename)).st_size == 0:
            featurize_file(file_path=os.path.join("splitted", filename),
                              output_path=output_path,
                              pretrained_model=model)
        print(output_path)
        print(counter, "is done")
        print("------")


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def counter(f):
    acc = 0
    if os.path.isdir(f):
        for file in os.listdir(f):
            if file.endswith('.csv') or file.endswith(".txt"):
                l = file_len(os.path.join(f, file))
                acc += l - 1
        return acc
    if os.path.isfile(f):
        if f.endswith('.csv') or f.endswith(".txt"):
            l = file_len(f)
            acc += l
        return (acc - 1)


if __name__ == "__main__":
    print(counter('multi_task_features_dmpnn_25_zinc_flagments'))
