import torch
from models.equireact import EquiReact
import torch.nn.functional as F


class AtomMapper(EquiReact):

    def forward(self, reactants_data, products_data):

        if self.sum_mode == 'node':
            predictor = self.score_predictor_nodes
        elif self.sum_mode == 'both':
            predictor  = self.score_predictor_nodes_with_edges
        else:
            raise NotImplementedError(f'sum mode "{self.sum_mode}" is not compatible with vector mode')

        x_react = self.forward_repr_mols(reactants_data)
        x_prod  = self.forward_repr_mols(products_data)

        if self.attention == 'cross':
            #x_react_mapped = [self.rp_attention(xp, xr, xr, need_weights=False)[0] for xp, xr in zip(x_prod, x_react)]
            #x_prod_mapped  = [self.rp_attention(xr, xp, xp, need_weights=False)[0] for xp, xr in zip(x_prod, x_react)]
            r2p_attention = [self.rp_attention(xr, xp, xp, need_weights=True)[1] for xp, xr in zip(x_prod, x_react)]

        elif self.attention == 'masked':
            def get_atoms(data):
                at = [g.x[:,[0]].to(torch.int)+1 for g in data if g.x.shape[0]>0]
                at = self.split_batch(at, data, merge=False)
                return at
            ratoms = get_atoms(reactants_data)
            patoms = get_atoms(products_data)
            #x_react_mapped = []
            #x_prod_mapped = []
            r2p_attention = []
            for xr, xp, ar, ap in zip(x_react, x_prod, ratoms, patoms):
                mask = (ap != ar.T).to(self.device)  # len(xp) × len(xr) ; True == no attention
                #x_react_mapped.append(self.rp_attention(xp, xr, xr, attn_mask=mask, need_weights=False)[0])
                #x_prod_mapped.append(self.rp_attention(xr, xp, xp, attn_mask=mask.T, need_weights=False)[0])
                r2p_attention.append(self.rp_attention(xr, xp, xp, attn_mask=mask.T, need_weights=True)[1])
        else:
            raise NotImplementedError(f'attention "{self.attention}" not defined')

        nat_list = [i.shape[0] for i in r2p_attention]
        nat_max = max(nat_list)
        for i, nat in enumerate(nat_list):
            r2p_attention[i] = F.pad(r2p_attention[i], (0, nat_max-nat, 0, nat_max-nat))[None,:,:]
        r2p_attention = torch.cat(r2p_attention)
        return r2p_attention

        #x = self.combine(torch.vstack(x_react_mapped), torch.vstack(x_prod_mapped))
        #batch = torch.sort(torch.hstack([g.batch for g in reactants_data])).values.to(self.device)
        #if self.graph_mode == 'energy':
        #    score_atom = predictor(x)
        #    score = scatter_add(score_atom, index=batch, dim=0)
        #elif self.graph_mode == 'vector':
        #    x = self.atom_diff_nonlin(x)
        #    x = scatter_add(x, index=batch, dim=0)
        #    score = predictor(x)
        #score = F.pad(score[:,:,None], (0, nat_max-1, 0, nat_max-1, 0, 0))
        #return score
