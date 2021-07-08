import os, statistics, csv, random

import pandas as pd


def create_balanced_kinase_data(file="kinase.tsv", length=90000):
    compounds = pd.read_csv("chembl27-bioactivities-less.tsv",
                            usecols=['Molecule ChEMBL ID', 'Target ChEMBL ID', 'Smiles'],
                            sep="\t")
    kinase = pd.read_csv(file, sep='\t', usecols=['ChEMBL ID'])['ChEMBL ID'].tolist()
    oxideroductase = pd.read_csv("oxidoreductase.tsv", sep='\t', usecols=['ChEMBL ID'])['ChEMBL ID'].tolist()
    transferase = pd.read_csv("transferase.tsv", sep='\t', usecols=['ChEMBL ID'])['ChEMBL ID'].tolist()

    # select positive set
    kinase_set= set()
    for index, row in compounds.iterrows():
        target = row["Target ChEMBL ID"]
        smiles = row['Smiles']
        if index % 10000 == 0:
            print(index)
        # find compounds that interact with proteins from kinase family
        if target in kinase:
            kinase_set.add(smiles)
        if len(kinase_set) > length / 2:
            break

    # select negative set
    oxideroductase_set = set()
    transferase_set = set()
    for index, row in compounds.iterrows():
        target = row["Target ChEMBL ID"]
        smiles = row['Smiles']
        if smiles not in kinase_set and target in oxideroductase:
            if len(oxideroductase_set) > length/4:
                continue
            oxideroductase_set.add(smiles)
        if smiles not in kinase_set and target in transferase:
            if len(transferase_set) > length/4:
                continue
            transferase_set.add(smiles)

    print("oxideroductase_set", len(oxideroductase_set))
    print("transferase_set", len(transferase_set))
    print("kinase_set", len(kinase_set))

    # label the positive and negative sets
    training_data = {}
    negative_set = transferase_set.union(oxideroductase_set)
    for key in kinase_set:
        training_data[key] = 1
    for key in negative_set:
        training_data[key] = 0

    print("training_data", len(training_data))

    with open('tri_family_kinase.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in training_data.items():
            writer.writerow([key, value])


def create_kinase_subfamilies_training_data(bioactivity_file="chembl27-bioactivities-less.tsv",
                                            ste_file="STE_proteins.tsv",
                                            tk_file="TK_proteins.tsv",
                                            length=3500):
    ste_proteins = pd.read_csv(ste_file, sep="\t")['ChEMBL ID'].tolist()
    tk_proteins = pd.read_csv(tk_file, sep="\t")['ChEMBL ID'].tolist()
    compounds = pd.read_csv(bioactivity_file,
                            usecols=['Molecule ChEMBL ID', 'Target ChEMBL ID', 'Smiles'],
                            sep="\t")
    ste_compounds = {}
    tk_compounds = {}

    for index, row in compounds.iterrows():
        compound = row["Molecule ChEMBL ID"]
        target = row["Target ChEMBL ID"]
        smiles = row["Smiles"]
        if index % 10000 == 0:
            print(index)
        if target in ste_proteins and compound not in ste_compounds.keys() and len(ste_compounds) < length:
            ste_compounds[smiles] = 1
        elif target in tk_proteins and compound not in tk_compounds.keys() and len(tk_compounds) < length:
            tk_compounds[smiles] = 0

    # if a compound interacts with both subfamilies, store it in a list to remove later
    to_be_deleted = []
    for ste_compound in ste_compounds.keys():
        for tk_compound in tk_compounds.keys():
            if ste_compound == tk_compound:
                to_be_deleted.append(ste_compound)

    # find the chembl ID's of common rows
    common_compound_chembls = {}
    for index, row in compounds.iterrows():
        chembl = row['Molecule ChEMBL ID']
        smiles = row['Smiles']
        target = row["Target ChEMBL ID"]
        if smiles in to_be_deleted:
            try:
                common_compound_chembls[chembl].append(target)
            except:
                common_compound_chembls[chembl] = target

    # write the common compound chembls to a file
    with open('ste_tk_common_chembls.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in common_compound_chembls.items():
            writer.writerow([key, value])

    # remove the elements in the list from both dictionaries
    print(len(to_be_deleted))
    for item in to_be_deleted:
        del ste_compounds[item]
        del tk_compounds[item]


    with open('kinase_subfamily_training.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in ste_compounds.items():
            writer.writerow([key, value])

    with open('kinase_subfamily_training.csv', 'a') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in tk_compounds.items():
            writer.writerow([key, value])


def create_regression_dataset(bioactivity_file="chembl27-bioactivities-less.tsv",
                                            protein_file="TK_proteins.tsv",
                                            length=1000000):

    tk_proteins = pd.read_csv(protein_file, sep="\t")['ChEMBL ID'].tolist()
    compounds = pd.read_csv(bioactivity_file,
                            usecols=['Molecule ChEMBL ID', 'Target ChEMBL ID', 'Smiles', 'Standard Value'],
                            sep="\t")

    target_bioactivity_mapping = {}
    for index, row in compounds.iterrows():
        target = row["Target ChEMBL ID"]
        smiles = row["Smiles"]
        standard_value = row['Standard Value']
        if index % 100000 == 0:
            print(index)
        if index > length:
            break
        if target in tk_proteins:
            try:
                target_bioactivity_mapping[smiles].append(standard_value)
            except:
                target_bioactivity_mapping[smiles] = [standard_value]

    for compound, bioactivities in target_bioactivity_mapping.items():
            target_bioactivity_mapping[compound] = statistics.median_low(bioactivities)

    # with open('tk_bioactivity_regression_dataset.csv', 'w') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for key, value in target_bioactivity_mapping.items():
    #         writer.writerow([key, value])
    return target_bioactivity_mapping


def create_multi_task_subfamily_training_data(file="kinase.tsv", length=90000):
    compounds = pd.read_csv("chembl27-bioactivities-less.tsv",
                            usecols=['Molecule ChEMBL ID', 'Target ChEMBL ID', 'Smiles'],
                            sep="\t")
    kinase = pd.read_csv(file, sep='\t', usecols=['ChEMBL ID'])['ChEMBL ID'].tolist()
    oxideroductase = pd.read_csv("oxidoreductase.tsv", sep='\t', usecols=['ChEMBL ID'])['ChEMBL ID'].tolist()
    transferase = pd.read_csv("transferase.tsv", sep='\t', usecols=['ChEMBL ID'])['ChEMBL ID'].tolist()

    # select positive set
    smiles_set = []
    kinase_labels = []
    oxideroductase_labels = []
    transferase_labels = []
    for index, row in compounds.iterrows():
        target = row["Target ChEMBL ID"]
        smiles = row['Smiles']
        if len(smiles_set) > length:
            break
        if index % 10000 == 0:
            print(index)
        if (target in kinase) or (target in oxideroductase) or (target in transferase):
            if target in kinase:
                kinase_labels.append(1)
            else:
                kinase_labels.append(0)
            if target in oxideroductase:
                oxideroductase_labels.append(1)
            else:
                oxideroductase_labels.append(0)
            if target in transferase:
                transferase_labels.append(1)
            else:
                transferase_labels.append(0)
            smiles_set.append(smiles)

    print("oxideroductase_set", len(oxideroductase_labels), sum(oxideroductase_labels))
    print("transferase_set", len(transferase_labels), sum(transferase_labels))
    print("kinase_set", len(kinase_labels), sum(kinase_labels))
    print("smiles_set", len(smiles_set))

    with open('multi_task_subfamily.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Smiles", "Kinase", "Oxideroductase", "Transferase"])
        writer.writerows(zip(smiles_set, kinase_labels, oxideroductase_labels, transferase_labels))


def create_active_inactive_dataset_for_single_protein(protein, bioactivity_file="chembl27-clean-bioactivites.tsv",
                                                       act_th=10, non_act_th=20, act_limit=-1):

    activity_values = pd.read_csv(bioactivity_file,
                        usecols=['Molecule ChEMBL ID', 'Target ChEMBL ID', 'Smiles', 'Standard Value'],
                        sep="\t")

    # abl: CHEMBL1862
    actives = {}
    inactives = {}
    unknowns = {} # to supplement inactives when necessary
    for index, row in activity_values.iterrows():
        if (index % 100000) == 0:
            print(index)
        if len(actives) == act_limit:
            break
        if row["Target ChEMBL ID"] == protein and not (str(row["Smiles"]) == "nan"):
            if row["Standard Value"] < act_th*1000:
                actives[row["Molecule ChEMBL ID"]] = row["Smiles"]
            elif row["Standard Value"] > non_act_th*1000:
                inactives[row["Molecule ChEMBL ID"]] = row["Smiles"]

    print("act, inact:", len(actives), len(inactives))

    # supplement inactives
    if len(inactives) < 60 * len(actives):
        for index, row in activity_values.iterrows():
            if (index % 100000) == 0:
                print(index)
            if not (row["Target ChEMBL ID"] == protein) and not (str(row["Smiles"]) == "nan"):
                if random.uniform(0, 1) < 0.2:
                    unknowns[row["Molecule ChEMBL ID"]] = row["Smiles"]

    # sample 60 times the size of actives
    selected_unknowns = dict(random.sample(unknowns.items(), (len(actives) * 60) - len(inactives)))
    for key, value in selected_unknowns.items():
        inactives[key] = value

    # remove duplicates
    tmp = list(actives.keys())
    for smiles in tmp:
        if smiles in inactives.keys():
            actives.pop(smiles)
            inactives.pop(smiles)

    with open(os.path.join("../data/chembl27-with-decoys/smiles", protein + '_actives_decoys_20k.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["ChEMBL", "Smiles", "Label"])
        for key, value in actives.items():
            writer.writerow([key, value, "1"])
        for key, value in inactives.items():
            writer.writerow([key, value, "0"])

    return actives, inactives


if __name__ == "__main__":
    reg = create_regression_dataset()