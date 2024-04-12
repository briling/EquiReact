#!/usr/bin/env python3

import sys
import argparse as ap
from itertools import compress
import random
import numpy as np
import pandas as pd
import chemprop
sys.path.insert(0, '../../../')
from process.splitter import split_dataset


def argparse():
    parser = ap.ArgumentParser()
    g1 = parser.add_mutually_exclusive_group(required=True)
    g1.add_argument('--true',          action='store_true', help='use true atom mapping')
    g1.add_argument('--rxnmapper',     action='store_true', help='use atom mapping from rxnmapper')
    g1.add_argument('--none',          action='store_true', help='use non-mapped smiles')
    g2 = parser.add_mutually_exclusive_group(required=True)
    g2.add_argument('-c', '--cyclo',   action='store_true', help='use curated Cyclo-23-TS dataset')
    g2.add_argument('-p', '--proparg', action='store_true', help='use Proparg-21-TS dataset with fragment-based SMILES')
    g2.add_argument('-g', '--gdb',     action='store_true', help='use curated GDB7-22-TS dataset')
    parser.add_argument('--scaffold',  action='store_true', help='use scaffold splits (random otherwise)')
    parser.add_argument('--withH',     action='store_true', help='use explicit H')
    args = parser.parse_args()
    return parser, args


if __name__ == "__main__":

    parser, args = argparse()
    if args.cyclo:
        data_path = '../../../data/cyclo/cyclo.csv'
        target_columns = 'G_act'
    elif args.gdb:
        data_path = '../../csv/gdb.csv'
        target_columns = 'dE0'
    elif args.proparg:
        data_path = '../../csv/proparg-fixarom.csv'
        target_columns = "Eafw"

    if args.rxnmapper:
        if args.withH:
            smiles_columns  ='rxn_smiles_rxnmapper_full'
        else:
            smiles_columns = 'rxn_smiles_rxnmapper'
    elif args.true:
        smiles_columns = 'rxn_smiles_mapped'
    elif args.none:
        smiles_columns = 'rxn_smiles'

    dataset = next(compress(('cyclo', 'gdb', 'proparg'), (args.cyclo, args.gdb, args.proparg)))
    config_path = f'../../data/hypers_{dataset}_cgr.json'


    CV = 10
    tr_frac = 0.8

    seed = 123
    scores = np.zeros(CV)
    for i in range(CV):
        print(f"CV iter {i+1}/{CV}")

        np.random.seed(seed)
        random.seed(seed)

        data = pd.read_csv(data_path, index_col=0)

        tr_indices, te_indices, val_indices, _ = split_dataset(nreactions=len(data),
                                                               splitter=('scaffold' if args.scaffold else 'random'),
                                                               tr_frac=tr_frac, dataset=dataset, subset=None)
        data.iloc[tr_indices].to_csv(f'data_{seed}_train.csv')
        data.iloc[te_indices].to_csv(f'data_{seed}_test.csv')
        data.iloc[val_indices].to_csv(f'data_{seed}_val.csv')

        arguments = [
            "--data_path",           f'data_{seed}_train.csv',
            "--separate_val_path",   f'data_{seed}_val.csv',
            "--separate_test_path",  f'data_{seed}_test.csv',
            "--dataset_type",        "regression",
            "--target_columns",      target_columns,
            "--smiles_columns",      smiles_columns,
            "--metric",              "mae",
            "--extra_metrics",       "rmse",
            "--epochs",              "300",
            "--reaction",
            "--config_path",         config_path,
            "--batch_size",          "50",
            "--save_dir",            f"fold_{i}"]
        if args.withH:
            arguments.append('--explicit_h')

        args_chemprop = chemprop.args.TrainArgs().parse_args(arguments)
        score, _ = chemprop.train.cross_validate(args=args_chemprop, train_func=chemprop.train.run_training)
        scores[i] = score

        seed += 1

    print("Mean score", scores.mean(), "std_score", scores.std())