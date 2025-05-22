import torch
import torch.nn as nn
import torch.nn.functional as F
from equivariant_diffusion.egnn_new import EGNN, GNN
import numpy as np
import subprocess
from torch_scatter import scatter_add, scatter_mean

class EGNNDynamics(nn.Module):
    def __init__(self, atom_nf, residue_nf, interh_nf, interhp_nf,
                 n_dims, joint_nf=16, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics',
                 norm_constant=0, inv_sublayers=2, sin_embedding=False,
                 normalization_factor=100, aggregation_method='sum',
                 update_pocket_coords=True, edge_cutoff_ligand=None,
                 edge_cutoff_pocket=None, edge_cutoff_interaction=None,
                 reflection_equivariant=True, edge_embedding_dim=None):
        super().__init__()
        self.mode = mode
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction
        self.edge_nf = edge_embedding_dim

        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, joint_nf)
        )

        self.atom_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, atom_nf)
        )

        self.residue_encoder = nn.Sequential(
            nn.Linear(residue_nf, 2 * residue_nf),
            act_fn,
            nn.Linear(2 * residue_nf, joint_nf)
        )

        self.residue_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * residue_nf),
            act_fn,
            nn.Linear(2 * residue_nf, residue_nf)
        )

        self.interh_encoder = nn.Sequential(
            nn.Linear(interh_nf, 2 * interh_nf),
            act_fn,
            nn.Linear(2 * interh_nf, joint_nf)
        )

        self.interh_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * interh_nf),
            act_fn,
            nn.Linear(2 * interh_nf, interh_nf)
        )

        self.interhp_encoder = nn.Sequential(
            nn.Linear(interhp_nf, 2 * interhp_nf),
            act_fn,
            nn.Linear(2 * interhp_nf, joint_nf)
        )

        self.interhp_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * interhp_nf),
            act_fn,
            nn.Linear(2 * interhp_nf, interhp_nf)
        )
        self.edge_embedding = nn.Embedding(3, self.edge_nf) \
            if self.edge_nf is not None else None
        self.edge_nf = 0 if self.edge_nf is None else self.edge_nf

        if condition_time:
            dynamics_node_nf = joint_nf + 1
        else:
            print('Warning: dynamics model is _not_ conditioned on time.')
            dynamics_node_nf = joint_nf

        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=dynamics_node_nf, in_edge_nf=self.edge_nf,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                reflection_equiv=reflection_equivariant
            )
            self.node_nf = dynamics_node_nf
            self.update_pocket_coords = update_pocket_coords

        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=dynamics_node_nf + n_dims, in_edge_nf=self.edge_nf,
                hidden_nf=hidden_nf, out_node_nf=n_dims + dynamics_node_nf,
                device=device, act_fn=act_fn, n_layers=n_layers,
                attention=attention, normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)

        self.device = device
        self.n_dims = n_dims
        self.condition_time = condition_time

    def forward(self, xh_atoms, xh_residues, xh_intersh, xh_intershp, t, mask_atoms, mask_residues, mask_intersh, mask_intershp):
        #print('=== dynamics =====')
        #print('ligand', xh_atoms)
        #print('interh', xh_intersh)
        #print('interhp', xh_intershp, flush=True)
        #print(t)
        x_atoms = xh_atoms[:, :self.n_dims].clone()
        h_atoms = xh_atoms[:, self.n_dims:].clone()

        x_residues = xh_residues[:, :self.n_dims].clone()
        h_residues = xh_residues[:, self.n_dims:].clone()

        x_intersh = xh_intersh[:, :self.n_dims].clone()
        h_intersh = xh_intersh[:, self.n_dims:].clone()
        x_intershp = xh_intershp[:, :self.n_dims].clone()
        h_intershp = xh_intershp[:, self.n_dims:].clone()

        #print(xh_atoms)
        # embed atom features and residue features in a shared space
        h_atoms = self.atom_encoder(h_atoms)
        h_residues = self.residue_encoder(h_residues)
        h_intersh = self.interh_encoder(h_intersh)
        h_intershp = self.interhp_encoder(h_intershp)

        # combine the two node types
        # ligand + pocket
        x = torch.cat((x_atoms, x_residues), dim=0)
        h = torch.cat((h_atoms, h_residues), dim=0)
        mask = torch.cat([mask_atoms, mask_residues])

        # ligand + h
        x2 = torch.cat((x_atoms, x_intersh), dim=0)
        h2 = torch.cat((h_atoms, h_intersh), dim=0)
        mask2 = torch.cat([mask_atoms, mask_intersh])

        # ligand + hp
        x3 = torch.cat((x_atoms, x_intershp), dim=0)
        h3 = torch.cat((h_atoms, h_intershp), dim=0)
        mask3 = torch.cat([mask_atoms, mask_intershp])

        #print('condition_time :', self.condition_time)
        #print('mask ', mask)
        #print('h :', h)
        #print(len(h))
        #print(len(h[0]))
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
                h_time_2 = torch.empty_like(h2[:, 0:1]).fill_(t.item())
                h_time_3 = torch.empty_like(h3[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t[mask]
                h_time_2 = t[mask2]
                h_time_3 = t[mask3]
            h = torch.cat([h, h_time], dim=1)
            h2 = torch.cat([h2, h_time_2], dim=1)
            h3 = torch.cat([h3, h_time_3], dim=1)
        #print('add t', h)
        #print(len(h))
        #print(len(h[0]))
        # get edges of a complete graph
        edges = self.get_edges(mask_atoms, mask_residues, x_atoms, x_residues)
        edges2 = self.get_edges(mask_atoms, mask_intersh, x_atoms, x_intersh)
        edges3 = self.get_edges(mask_atoms, mask_intershp, x_atoms, x_intershp)
        #print('edges :',edges)
        #print('edges :',len(edges))
        
        #print('edges2 :',edges2)
        #print('edges2 :',len(edges2))
        
        #print('mask_inters',mask_inters)
        assert torch.all(mask[edges[0]] == mask[edges[1]])

        # Get edge types
        # edge_nf=0 : through this part
        if self.edge_nf > 0:
            # 0: ligand-pocket, 1: ligand-ligand, 2: pocket-pocket
            edge_types = torch.zeros(edges.size(1), dtype=int, device=edges.device)
            edge_types[(edges[0] < len(mask_atoms)) & (edges[1] < len(mask_atoms))] = 1
            edge_types[(edges[0] >= len(mask_atoms)) & (edges[1] >= len(mask_atoms))] = 2

            # Learnable embedding
            edge_types = self.edge_embedding(edge_types)
        else:
            edge_types = None

        if self.mode == 'egnn_dynamics':
            #print('==egnn_dynamics start ==')
            update_coords_mask = None if self.update_pocket_coords \
                else torch.cat((torch.ones_like(mask_atoms),
                                torch.zeros_like(mask_residues))).unsqueeze(1)
            update_coords_mask_2 = None if self.update_pocket_coords \
                else torch.cat((torch.ones_like(mask_atoms),
                                torch.zeros_like(mask_intersh))).unsqueeze(1)
            update_coords_mask_3 = None if self.update_pocket_coords \
                else torch.cat((torch.ones_like(mask_atoms),
                                torch.zeros_like(mask_intershp))).unsqueeze(1)

            h_final, x_final = self.egnn(h, x, h2, x2, h3, x3, edges, edges2, edges3,
                                                mask_atom=mask_atoms, mask_intersh=mask_intersh, mask_intershp=mask_intershp,
                                                update_coords_mask=update_coords_mask,
                                                update_coords_mask_2=update_coords_mask_2,
                                                update_coords_mask_3=update_coords_mask_3,
                                                batch_mask=mask, batch_mask_2=mask2, batch_mask_3=mask3, edge_attr=edge_types)

            vel = (x_final - x)

        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=None, edge_attr=edge_types)
            vel = output[:, :3]
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        # decode atom and residue features
        h_final_atoms = self.atom_decoder(h_final[:len(mask_atoms)])
        h_final_residues = self.residue_decoder(h_final[len(mask_atoms):])

        if torch.any(torch.isnan(vel)):
            if self.training:
                vel[torch.isnan(vel)] = 0.0
            else:
                raise ValueError("NaN detected in EGNN output")

        
        if self.update_pocket_coords:
            # in case of unconditional joint distribution, include this as in
            # the original code
            vel = remove_mean_batch(vel, mask)

        return torch.cat([vel[:len(mask_atoms)], h_final_atoms], dim=-1), \
               torch.cat([vel[len(mask_atoms):], h_final_residues], dim=-1)

    def get_edges(self, batch_mask_ligand, batch_mask_pocket, x_ligand, x_pocket):
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]

        if self.edge_cutoff_l is not None:
            adj_ligand = adj_ligand & (torch.cdist(x_ligand, x_ligand) <= self.edge_cutoff_l)

        if self.edge_cutoff_p is not None:
            adj_pocket = adj_pocket & (torch.cdist(x_pocket, x_pocket) <= self.edge_cutoff_p)

        if self.edge_cutoff_i is not None:
            adj_cross = adj_cross & (torch.cdist(x_ligand, x_pocket) <= self.edge_cutoff_i)

        adj = torch.cat((torch.cat((adj_ligand, adj_cross), dim=1),
                         torch.cat((adj_cross.T, adj_pocket), dim=1)), dim=0)

        
        edges = torch.stack(torch.where(adj), dim=0)

        return edges
    
       
    def remove_mean_batch(cls, x_lig, x_pocket, x_inter, lig_indices, pocket_indices, inter_indices):

        # Just subtract the center of mass of the sampled part
        mean = scatter_mean(x_lig, lig_indices, dim=0)

        x_lig = x_lig - mean[lig_indices]
        x_pocket = x_pocket - mean[pocket_indices]
        if len(x_inter)!=0: x_inter = x_inter -  mean[inter_indices]
        return x_lig, x_pocket, x_inter