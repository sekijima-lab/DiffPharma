## Dependencies

### Cuda environment
| Software     | Version |
|--------------|---------|
| CUDA         | 11.8    |
| cudnn        | 8.9.7   |

### conda environment
```bash
conda env create -n Int-env -f environment.yml
```

| Software          | Version   |
|-------------------|-----------|
| Python            | 3.10.4    |
| numpy             | 1.22.3    |
| PyTorch           | 2.1.1     |
| PyTorch cuda      | 11.8      |
| Torchvision       | 0.16.1    |
| Torchaudio        | 2.1.1     |
| PyTorch Scatter   | 2.1.2     |
| PyTorch Lightning | 1.7.4     |
| RDKit             | 2022.03.2 |
| WandB             | 0.13.1    |
| BioPython         | 1.79      |
| imageio           | 2.21.2    |
| SciPy             | 1.7.3     |
| OpenBabel         | 3.1.1     |
| ODDT              | 0.7       |



### Data download
Download the training, validation and test datasets: [Data]()

```bash
tar xvzf DiffInt_crossdock_data.tar.gz
```

### Data construction by yourself
(You don't need to construct data by yourself.)
Download and extract the dataset as described by the authors of [Pocket2Mol](https://github.com/pengxingang/Pocket2Mol/tree/main/data).  
Download the dataset archive `crossdocked_pocket10.tar.gz` and the split file `split_by_name.pt` to `data` directory.
```bash
.
├── data
│   ├── DiffInt_crossdock_data.tar.gz
│   └── split_by_name.pt
```
Extract the TAR archive
```bash
tar -xzvf crossdocked_pocket10.tar.gz
```

data preparation step 1
```bash
python process_crossdock.py /data/directory/path/ --outdir /output/directory/path/
```
For example
```bash
python process_crossdock.py data/ --outdir data/crossdocked/
```

data preparation step 2: add interaction information
```bash
python interaction_construct.py --data_dir /step_1/directory/path/ --out_dir /step_2/directory/path/ --pdb_dir /pdb_data/directory/path/
```
For example
```bash
python interaction_construct.py --data_dir data/crossdock/ --out_dir data/crossdocked_interaction/ --pdb_dir data/crossdocked_pocket10/
```

### Training
```bash
python -u train.py --config config/DiffPharma.yml
```

### Molecule generation
Download the pretrained model:[Model]().
Generation of 100 ligand molecules for 100 protein pockets.

```bash
python test_npz.py --checkpoint checkpoint_file --test_dir /data/directory/path/ --outdir /out/directory/path/
```
For example
```bash
python test_npz.py --checkpoint checkpoints/DiffPharma_best.ckpt --test_dir DiffInt_crossdock_data/ --outdir sample
```

Generated molecules used in the paper are ```example/DiffInt_generated_molecules.tar.gz ```


### Generate 10 ligand molecules for one pocket
```bash
python test_single.py --checkpoint checkpoint_file --outdir /out/directory/path/ --pdb /pdb/file/path/ --sdf /sdf/file/path/
```

Or you can use Google Colabratory.(This notebook has been confirmed to work on May 19, 2025.)

```bash
.
├── colab
│   └── DiffPharma_generate.ipynb
```
>>>>>>> first upload
