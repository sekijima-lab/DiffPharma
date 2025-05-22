import oddt
import numpy as np
from oddt.interactions import hbonds, hbond_acceptor_donor,  hydrophobic_contacts
from tqdm import tqdm
import argparse
from pathlib import Path

def hbond_create(protein, ligand):
    li_do = hbond_acceptor_donor(protein,ligand)
    li_ac = hbond_acceptor_donor(ligand, protein)

    # ligand pocket interaction id
    po_id_d = [li_do[0][i]['id'] for i in range(len(li_do[0])) if li_do[2][i]==True]
    po_id_a = [li_ac[1][i]['id'] for i in range(len(li_ac[0])) if li_ac[2][i]==True]
 
    li_id_d = [li_do[1][i]['id'] for i in range(len(li_do[0])) if li_do[2][i]==True]
    li_id_a = [li_ac[0][i]['id'] for i in range(len(li_ac[0])) if li_ac[2][i]==True]

    int_id = []
    if len(li_id_d)>0:
        [int_id.append([li_id_d[i],0,po_id_d[i],0])for i in range(len(li_id_d))]
    if len(li_id_a)>0:
        [int_id.append([0,li_id_a[i],0,po_id_a[i]])for i in range(len(li_id_a))]
    
    # interaction particle coords
    coords_lido_1 = [(3*li_do[0][i]['coords']+li_do[1][i]['coords'])/4 for i in range(len(li_do[0])) if li_do[2][i]==True]
    coords_lido_2 = [(li_do[0][i]['coords']+li_do[1][i]['coords'])/2 for i in range(len(li_do[0])) if li_do[2][i]==True]
    coords_lido_3 = [(li_do[0][i]['coords']+3*li_do[1][i]['coords'])/4 for i in range(len(li_do[0])) if li_do[2][i]==True]
    coords_liac_1 = [(3*li_ac[0][i]['coords']+li_ac[1][i]['coords'])/4 for i in range(len(li_ac[0])) if li_ac[2][i]==True]
    coords_liac_2 = [(li_ac[0][i]['coords']+li_ac[1][i]['coords'])/2 for i in range(len(li_ac[0])) if li_ac[2][i]==True]
    coords_liac_3 = [(li_ac[0][i]['coords']+3*li_ac[1][i]['coords'])/4 for i in range(len(li_ac[0])) if li_ac[2][i]==True]

    inter_coords=[]
    [inter_coords.append(item) for pair in zip(coords_lido_1, coords_lido_2, coords_lido_3) for item in pair]
    [inter_coords.append(item) for pair in zip(coords_liac_1, coords_liac_2, coords_liac_3) for item in pair]
    inter_coords=np.array(inter_coords)

    #len_do = len(coords_lido_1)*3
    #len_ac = len(coords_liac_1)*3
    #do_ac_id = np.concatenate([np.zeros(len_do,dtype=int),np.ones(len_ac,dtype=int)])
    do_ids = np.tile([0, 1, 1], len(coords_lido_1))
    ac_ids = np.tile([0, 2, 2], len(coords_liac_1))
    do_ac_id = np.concatenate([do_ids, ac_ids])
    inter_one_hot = np.identity(3)[do_ac_id]

    if len(int_id)==0:
        int_id=np.array([])

    return int_id, inter_coords, inter_one_hot

def hydrophobic_data(protein, ligand):
    hydo = hydrophobic_contacts(protein, ligand)

    # ligand pocket interaction id
    int_id = np.stack((hydo[1]['id'], hydo[0]['id']), axis=1).astype(np.int64)
    
    # interaction particle coords
    C1 = (2*hydo[0]['coords'] + hydo[1]['coords'])/3
    C2 = (hydo[0]['coords'] + 2*hydo[1]['coords'])/3
    coords = np.array([item for pair in zip(C1,C2) for item in pair])

    # ligand atom type
    mapping = {'SP':0, 'C.ar': 1,'C.3': 2,'C.2': 3,'C.1': 4} # others:5
    atom_type = [x['atomtype'] for x in hydo[1]]
    atom_sp = ['SP']*len(atom_type)
    atom_label = [x for i in range(len(atom_type)) for x in (atom_sp[i], atom_type[i])]
    atom_int = [mapping.get(x, 5) for x in atom_label]
    one_hot = np.identity(6)[atom_int]

    return int_id, coords, one_hot

def data_create(file, base_path):
    inter_id_h = []
    inter_mask_h = []
    inter_coords_h = []
    inter_one_hot_h = []
    inter_id_hp = []
    inter_mask_hp = []
    inter_coords_hp = []
    inter_one_hot_hp = []

    pdb_name = [x.split('.pdb_')[0]+'.pdb' for x in file['names']]
    lig_name = [x.split('.pdb_')[1] for x in file['names']]

    for i in tqdm(range(len(file['names']))):
        protein = next(oddt.toolkit.readfile('pdb',str(Path(base_path, pdb_name[i]))))
        ligand = next(oddt.toolkit.readfile('sdf',str(Path(base_path, lig_name[i]))))
        protein.protein = True
        ligand.removeh()

        h_id, h_coords, h_one_hot = hbond_create(protein, ligand)
        hp_id, hp_coords, hp_one_hot = hydrophobic_data(protein, ligand)
        [inter_id_h.append(x) for x in h_id]
        [inter_coords_h.append(x) for x in h_coords]
        [inter_one_hot_h.append(x)for x in h_one_hot]
        [inter_mask_h.append(x) for x in np.full(len(h_coords),i)]
        [inter_id_hp.append(x) for x in hp_id]
        [inter_coords_hp.append(x) for x in hp_coords]
        [inter_one_hot_hp.append(x)for x in hp_one_hot]
        [inter_mask_hp.append(x) for x in np.full(len(hp_coords),i)]

    inter_id_h = np.array(inter_id_h)
    inter_coords_h = np.array(inter_coords_h)
    inter_one_hot_h = np.array(inter_one_hot_h)
    inter_mask_h = np.array(inter_mask_h)
    inter_id_hp = np.array(inter_id_hp)
    inter_coords_hp = np.array(inter_coords_hp)
    inter_one_hot_hp = np.array(inter_one_hot_hp)
    inter_mask_hp = np.array(inter_mask_hp)

    return inter_id_h, inter_coords_h, inter_one_hot_h, inter_mask_h, inter_id_hp, inter_coords_hp, inter_one_hot_hp, inter_mask_hp

def data_save(data_dir, new_dir, pdb_dir, data_type):
    print('=== data creation for ' + data_type + ' ===')

    # read original data without hbond
    file_path = Path(data_dir,data_type + '.npz')
    data_file = np.load(file_path)

    inter_id_h, inter_coords_h, inter_one_hot_h, inter_mask_h, inter_id_hp, inter_coords_hp, inter_one_hot_hp, inter_mask_hp = data_create(data_file, pdb_dir)

    # data save
    out_name = Path(new_dir, data_type +'.npz')

    np.savez(out_name,
            names=data_file['names'],
            lig_coords=data_file['lig_coords'],
            lig_one_hot=data_file['lig_one_hot'],
            lig_mask=data_file['lig_mask'],
            pocket_coords=data_file['pocket_coords'],
            pocket_one_hot=data_file['pocket_one_hot'],
            pocket_mask=data_file['pocket_mask'],
            interh_id=inter_id_h,
            interh_coords = inter_coords_h,
            interh_one_hot = inter_one_hot_h,
            interh_mask=inter_mask_h,
            interhp_id=inter_id_hp,
            interhp_coords = inter_coords_hp,
            interhp_one_hot = inter_one_hot_hp,
            interhp_mask=inter_mask_hp,
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path,default=None)
    parser.add_argument('--out_dir', type=Path,default=None)
    parser.add_argument('--pdb_dir', type=Path,default=None)
    parser.add_argument('--skip_existing', action='store_true', default=True)
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=args.skip_existing)

    data_type = ['val', 'test', 'train']
    [data_save(args.data_dir, args.out_dir ,args.pdb_dir ,n) for n in data_type]

if __name__ == "__main__":
    main()