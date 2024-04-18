import os
from os.path import exists, join
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from process.create_graph import get_graph, reader


class Reiher(Dataset):

    def __init__(self, process=True,
                 processed_dir='data/reiher/processed/',
                 noH=True, atom_mapping=False):

        self.version = 1  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.max_number_of_reactants = 1
        self.max_number_of_products = 1
        self.processed_dir = processed_dir + '/'
        self.atom_mapping = atom_mapping
        self.noH = noH
        target_column = 'barrier_kcal/mol'

        csv_path='data/reiher/reiher.csv'
        self.files_dir='data/reiher/xyz2'

        dataset_prefix = os.path.splitext(os.path.basename(csv_path))[0]
        if noH:
            dataset_prefix += '.noH'
        self.paths = SimpleNamespace(
                rg = join(self.processed_dir, f'{dataset_prefix}.v{self.version}.reactants_graphs.pt'),
                pg = join(self.processed_dir, f'{dataset_prefix}.v{self.version}.products_graphs.pt'),
                mp = join(self.processed_dir, f'{dataset_prefix}.v{self.version}.p2r_mapping.pt'),
                )

        print("Loading data into memory...")
        print(f'{dataset_prefix=}')

        self.df = pd.read_csv(csv_path)
        self.nreactions = len(self.df)
        self.indices = np.arange(self.nreactions)
        self.labels = torch.tensor(self.df[target_column].values)

        if process == True:
            print("Processing by request...")
            self.process()
        else:
            if exists(self.paths.rg) and exists(self.paths.pg) and exists(self.paths.mp):
                self.reactants_graphs = torch.load(self.paths.rg)
                self.products_graphs = torch.load(self.paths.pg)
                self.p2r_maps        = torch.load(self.paths.mp)
                print(f"Coords and graphs successfully read from {self.processed_dir}")
            else:
                print("Processed data not found, processing data...")
                self.process()

        self.standardize_labels()


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        r = self.reactants_graphs[idx]
        p = self.products_graphs[idx]
        label = self.labels[idx]
        if self.atom_mapping:
            return label, idx, r, p, self.p2r_maps[idx]
        else:
            return label, idx, r, p


    def process(self):

        print(f"Processing xyz files and saving coords to {self.processed_dir}")
        if not exists(self.processed_dir):
            os.mkdir(self.processed_dir)
            print(f"Creating processed directory {self.processed_dir}")

        self.products_graphs = []
        self.reactants_graphs = []
        self.p2r_maps = []

        for idx in tqdm(self.indices, desc="making graphs"):

            r_atomtypes, r_coords = reader(f'{self.files_dir}/{self.df.reactant_xyz[idx]}.xyz')
            p_atomtypes, p_coords = reader(f'{self.files_dir}/{self.df.product_xyz[idx]}.xyz')
            assert len(r_atomtypes) == len(p_atomtypes), f'{idx}'
            assert len(r_coords) == len(r_atomtypes), f'{idx}'
            assert len(p_coords) == len(p_atomtypes), f'{idx}'

            rgraph, ratoms, rmap = self.make_graph(r_atomtypes, r_coords,  f'r{idx}', idx)
            pgraph, patoms, pmap = self.make_graph(p_atomtypes, p_coords,  f'p{idx}', idx)
            self.reactants_graphs.append(rgraph)
            self.products_graphs.append(pgraph)

            assert np.all(sorted(rmap)==np.arange(len(rmap))), f'atoms missing from mapping {idx}'
            assert np.all(sorted(rmap)==sorted(pmap)), f'atoms missing from mapping {idx}'
            p2rmap = np.hstack([np.where(pmap==j)[0] for j in rmap])
            assert np.all(rmap == pmap[p2rmap])
            assert np.all(ratoms == patoms[p2rmap]), f'{idx}'
            self.p2r_maps.append(p2rmap)

        torch.save(self.reactants_graphs, self.paths.rg)
        torch.save(self.products_graphs, self.paths.pg)
        torch.save(self.p2r_maps, self.paths.mp)
        print(f"Saved graphs to {self.paths.rg} and {self.paths.pg}")


    def make_graph(self, atoms, coords, ireact, idx, smi2=None):

        smi = '.'.join([f'[{a}:{i+1}]' for i, a in enumerate(atoms)])
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        assert mol is not None, f"mol obj {ireact} is None from smi {smi}"
        Chem.SanitizeMol(mol)

        if self.noH:
            mol = Chem.RemoveAllHs(mol, sanitize=False)
            Chem.SanitizeMol(mol)
            noH_idx = np.where(atoms!='H')
            new_atoms = atoms[noH_idx]
            new_coords = coords[noH_idx]
        else:
            new_atoms = atoms
            new_coords = coords

        atom_map = np.array([at.GetAtomMapNum() for at in mol.GetAtoms()])
        atom_map = atom_map.argsort().argsort()  # elements rank

        new_atoms = new_atoms[atom_map]
        new_coords = new_coords[atom_map]
        graph = get_graph(mol, new_atoms, new_coords, idx)

        return graph, new_atoms, atom_map


    def standardize_labels(self):
        mean = torch.mean(self.labels)
        std = torch.std(self.labels)
        self.std = std
        self.labels = (self.labels - mean)/std
