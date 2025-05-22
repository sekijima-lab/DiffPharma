import argparse
import warnings
from pathlib import Path
from time import time
import numpy as np
import os

import torch
from rdkit import Chem
from tqdm import tqdm
import oddt

from lightning_modules import LigandPocketDDPM
from analysis.molecule_builder import build_molecule, process_molecule
from dataset import ProcessedLigandPocketDataset
import utils
from constants import dataset_params, FLOAT_TYPE, INT_TYPE
from equivariant_diffusion.conditional_model import ConditionalDDPM
from torch_scatter import scatter_add, scatter_mean
from process_crossdock import process_ligand_and_pocket
from interaction_construct import hbond_create, hydrophobic_data


def ligand_generation(test_file, checkpoint, outdir=None, batch_size=6, n_samples=10, relax=True, all_frags=True, save=True):
    #t_pocket_start = time()      
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_dims = 3
    model = LigandPocketDDPM.load_from_checkpoint(checkpoint, map_location=device)
    model = model.to(device)

    test_dataset = ProcessedLigandPocketDataset(test_file, center=False)

    if save:
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True)
        raw_sdf_dir = Path(outdir, 'raw')
        raw_sdf_dir.mkdir(exist_ok=True)
        processed_sdf_dir = Path(outdir, 'processed')
        processed_sdf_dir.mkdir(exist_ok=True)
        sdf_out_file_raw = Path(raw_sdf_dir, test_dataset[0]['names']+'_gen.sdf')
        sdf_out_file_processed = Path(processed_sdf_dir, test_dataset[0]['names']+'_gen.sdf')

    pocket_coords = test_dataset[0]['pocket_coords'].repeat(batch_size,1)
    pocket_one_hot = test_dataset[0]['pocket_one_hot'].repeat(batch_size,1)
    pocket_mask = torch.repeat_interleave(
            torch.arange(batch_size, device=device, dtype=INT_TYPE),len(test_dataset[0]['pocket_mask'])
        )
    pocket_size = torch.tensor([test_dataset[0]['num_pocket_nodes']]*batch_size, device=device, dtype=INT_TYPE)

    pocket ={
        'x': pocket_coords.to(device, FLOAT_TYPE),
        'one_hot': pocket_one_hot.to(device, FLOAT_TYPE),
        'size': pocket_size.to(device, INT_TYPE),
        'mask': pocket_mask.to(device, INT_TYPE)
        }
    
    interh_coords = test_dataset[0]['interh_coords'].repeat(batch_size,1)
    interh_one_hot = test_dataset[0]['interh_one_hot'].repeat(batch_size,1)
    interh_mask = torch.repeat_interleave(
            torch.arange(batch_size, device=device, dtype=INT_TYPE),len(test_dataset[0]['interh_mask'])
        )
    interh_size = torch.tensor([test_dataset[0]['num_interh_nodes']]*batch_size, device=device, dtype=INT_TYPE)

    interh ={
        'x': interh_coords.to(device, FLOAT_TYPE),
        'one_hot': interh_one_hot.to(device, FLOAT_TYPE),
        'size': interh_size.to(device, INT_TYPE),
        'mask': interh_mask.to(device, INT_TYPE)
        }
    
    interhp_coords = test_dataset[0]['interhp_coords'].repeat(batch_size,1)
    interhp_one_hot = test_dataset[0]['interhp_one_hot'].repeat(batch_size,1)
    interhp_mask = torch.repeat_interleave(
            torch.arange(batch_size, device=device, dtype=INT_TYPE),len(test_dataset[0]['interhp_mask'])
        )
    interhp_size = torch.tensor([test_dataset[0]['num_interhp_nodes']]*batch_size, device=device, dtype=INT_TYPE)

    interhp ={
        'x': interhp_coords.to(device, FLOAT_TYPE),
        'one_hot': interhp_one_hot.to(device, FLOAT_TYPE),
        'size': interhp_size.to(device, INT_TYPE),
        'mask': interhp_mask.to(device, INT_TYPE)
        }
    
    # Pocket's center of mass
    pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0)

    num_nodes_lig = None
    if num_nodes_lig is None:
        num_nodes_lig = model.ddpm.size_distribution.sample_conditional(n1=None, n2=pocket['size'])
    
    # if you want to fix ligand node size
    #num_nodes_lig = torch.full((batch_size,), 33, device=device)

    #print('num_nodes_lig:',num_nodes_lig)


    # statistical data
    all_molecules = []
    valid_molecules = []
    processed_molecules = []
    n_generated = 0
    n_valid = 0
    iter=0

    while len(valid_molecules) < n_samples:
        iter +=1

        if type(model.ddpm) == ConditionalDDPM: 
            print(type(model.ddpm))
            xh_lig, xh_pocket, lig_mask, pocket_mask = model.ddpm.sample_given_pocket(pocket, interh, interhp, num_nodes_lig,timesteps=None)

        pocket_com_after = scatter_mean(xh_pocket[:, :x_dims], pocket_mask, dim=0)
        xh_pocket[:, :x_dims] += \
                 (pocket_com_before - pocket_com_after)[pocket_mask]
        xh_lig[:, :x_dims] += \
                (pocket_com_before - pocket_com_after)[lig_mask]
        
        # Build mol objects
        x = xh_lig[:, :x_dims].detach().cpu()
        atom_type = xh_lig[:, x_dims:].argmax(1).detach().cpu()
        lig_mask = lig_mask.cpu()
        
        dataset_info = dataset_params['full_hDSP_hpSSP']

        mols_batch = []
        for mol_pc in zip(utils.batch_to_list(x, lig_mask),
                            utils.batch_to_list(atom_type, lig_mask)):

            mol = build_molecule(*mol_pc, dataset_info, add_coords=True)
            mol = process_molecule(mol,
                                    add_hydrogens=False,
                                    sanitize=True,
                                    relax_iter=0,
                                    largest_frag=False)
            if mol is not None:
                mols_batch.append(mol)

        all_molecules.extend(mols_batch)
        # Filter to find valid molecules
        mols_batch_processed = [
                process_molecule(m, sanitize=True,
                                relax_iter=(200 if relax else 0),
                                largest_frag=not all_frags)
                for m in mols_batch
            ]
        processed_molecules.extend(mols_batch_processed)
        valid_mols_batch = [m for m in mols_batch_processed if m is not None]
        
        n_generated += batch_size
        n_valid += len(valid_mols_batch)
        valid_molecules.extend(valid_mols_batch)

    valid_molecules = valid_molecules[:n_samples]

    # Reorder raw files
    all_molecules = \
        [all_molecules[i] for i, m in enumerate(processed_molecules)
        if m is not None] + \
        [all_molecules[i] for i, m in enumerate(processed_molecules)
        if m is None]

    if save:
        # Write SDF files
        utils.write_sdf_file(sdf_out_file_raw, all_molecules)
        utils.write_sdf_file(sdf_out_file_processed, valid_molecules)


    return valid_molecules

def process_data(sdf_file, pdb_file):
    dataset_info = dataset_params['full_hDSP_hpSSP']
    amino_acid_dict = dataset_info['aa_encoder']
    atom_dict = dataset_info['atom_encoder']
    atom_decoder = dataset_info['atom_decoder']
    dist_cutoff=8.0
    ligand_data, pocket_data = process_ligand_and_pocket(pdb_file,
                                                          sdf_file,
                                                          atom_dict=atom_dict, dist_cutoff=dist_cutoff,amino_acid_dict=amino_acid_dict, ca_only=False)

    lig_mask = np.zeros(len(ligand_data['lig_coords']))
    pocket_mask = np.zeros(len(pocket_data['pocket_coords']))
    file_name = np.array([Path(pdb_file).stem])

    np.savez('tmp1.npz',
        names = file_name,
        lig_coords=ligand_data['lig_coords'],
        lig_one_hot=ligand_data['lig_one_hot'],
        lig_mask = lig_mask,
        pocket_coords=pocket_data['pocket_coords'],
        pocket_one_hot=pocket_data['pocket_one_hot'],
        pocket_mask = pocket_mask
        )
        
def process_data_h(sdf_file, pdb_file):
    test_npz = np.load('tmp1.npz')
    protein = next(oddt.toolkit.readfile('pdb',pdb_file))
    ligand = next(oddt.toolkit.readfile('sdf',sdf_file))
    protein.protein = True
    ligand.removeh()

    h_id, h_coords, h_one_hot = hbond_create(protein, ligand)
    hp_id, hp_coords, hp_one_hot = hydrophobic_data(protein, ligand)
    h_mask = [x for x in np.full(len(h_coords),0)]
    hp_mask = [x for x in np.full(len(hp_coords),0)]

    np.savez('tmp2.npz',
        names = test_npz['names'],
        lig_coords=test_npz['lig_coords'],
        lig_one_hot=test_npz['lig_one_hot'],
        lig_mask = test_npz['lig_mask'],
        pocket_coords = test_npz['pocket_coords'],
        pocket_one_hot= test_npz['pocket_one_hot'],
        pocket_mask = test_npz['pocket_mask'],
        interh_id = h_id,
        interh_coords = h_coords,
        interh_one_hot = h_one_hot,
        interh_mask = h_mask,
        interhp_id = hp_id,
        interhp_coords = hp_coords,
        interhp_one_hot = hp_one_hot,
        interhp_mask = hp_mask
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path,default=None)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--pdb', type=str,default=None)
    parser.add_argument('--sdf', type=str,default=None)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--all_frags', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--resamplings', type=int, default=10)
    parser.add_argument('--jump_length', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument('--fix_n_nodes', action='store_true')
    parser.add_argument('--n_nodes_bias', type=int, default=0)
    parser.add_argument('--n_nodes_min', type=int, default=0)
    parser.add_argument('--skip_existing', action='store_true',default=True)
    args = parser.parse_args()

    
    process_data(sdf_file=args.sdf, pdb_file=args.pdb)
    process_data_h(sdf_file=args.sdf, pdb_file=args.pdb) 
    ligand_generation(outdir=args.outdir, test_file='tmp2.npz', checkpoint=args.checkpoint, save=True)


if __name__ == "__main__":
    main()