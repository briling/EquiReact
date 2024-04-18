import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

df = pd.read_csv(f'reiher.csv')

def xyz2mol(xyz, charge=0):
    raw_mol = Chem.MolFromXYZFile(xyz)
    mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    return mol

for i in range(len(df[:10])):
    print(i)
    r = f'xyz2/{df.iloc[i].reactant_xyz}.xyz'
    p = f'xyz2/{df.iloc[i].product_xyz}.xyz'

    print(r)
    rm = xyz2mol(r)
    print(p)
    pm = xyz2mol(p, charge=+1)
    #pm = xyz2mol(p)
    #print(Chem.MolToSmiles(rm))
    #print(Chem.MolToSmiles(pm))



