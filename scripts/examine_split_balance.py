import os
import pickle
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data


BASE = '/data/rsg/chemistry/yangk/lsc_experiments_dump_splits/data'
DATASETS = ['pcba', 'muv', 'hiv', 'bace', 'bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'chembl']


def compute_ratios(data: MoleculeDataset) -> np.ndarray:
    ratios = np.nanmean(np.array(data.targets(), dtype=np.float), axis=0)
    ratios = np.maximum(ratios, 1 - ratios)

    return ratios


def examine_split_balance():
    for dataset in DATASETS:
        print(dataset)

        # Load task names for the dataset
        data_path = os.path.join(BASE, dataset, f'{dataset}.csv')
        data = get_data(data_path)

        # Get class balance ratios for full dataset
        ratios = compute_ratios(data)

        # Initialize array of diffs between ratios
        ratio_diffs = []

        # Loop through folds
        for fold in os.listdir(os.path.join(BASE, dataset, 'scaffold')):
            # Open fold indices
            with open(os.path.join(BASE, dataset, 'scaffold', fold, '0', 'split_indices.pckl'), 'rb') as f:
                indices = pickle.load(f)

            # Get test data
            test_data = MoleculeDataset([data[index] for index in indices[2]])

            # Get test ratios
            test_ratios = compute_ratios(test_data)

            # Compute ratio diff
            ratio_diff = np.maximum(ratios / test_ratios, test_ratios / ratios)

            # Add ratio diff
            ratio_diffs.append(ratio_diff)

        # Convert to numpy array
        ratio_diffs = np.array(ratio_diffs)  # num_folds x num_tasks

        # Determine number of folds and number of failures
        num_folds = len(ratio_diffs)
        num_failures = np.sum(np.isnan(ratio_diffs))

        # Average across tasks
        ratio_diffs = np.nanmean(ratio_diffs, axis=1)  # num_folds

        # Compute mean and standard deviation across folds
        mean, std = np.nanmean(ratio_diffs), np.nanstd(ratio_diffs)

        print(f'mean = {mean}, std = {std}, num_folds = {num_folds}, num_failures = {num_failures}')


if __name__ == '__main__':
    examine_split_balance()