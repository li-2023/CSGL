import math
import torch
from dgl.nn import GraphConv, GATConv, SAGEConv, SGConv, TAGConv
from dgl.nn.pytorch.glob import SumPooling
from torch.nn import ModuleList
from torch.nn.functional import one_hot
import torch.nn as nn
import torch.nn.functional as F


class GNN(torch.nn.Module):
    def __init__(self, args, feature_len):
        super(GNN, self).__init__()
        self.gnn = args.gnn
        self.n_layer = args.layer
        self.feature_len = feature_len
        self.dim = args.dim
        self.gnn_layers = ModuleList([])
        if self.gnn in ['gcn', 'gat', 'sage', 'tag']:
            for i in range(self.n_layer):
                if self.gnn == 'gcn':
                    self.gnn_layers.append(GraphConv(in_feats=feature_len if i == 0 else self.dim,
                                                     out_feats=self.dim,
                                                     activation=None if i == self.n_layer - 1 else torch.relu))
                elif self.gnn == 'gat':
                    num_heads = 16  
                    self.gnn_layers.append(GATConv(in_feats=feature_len if i == 0 else self.dim,
                                                   out_feats=self.dim // num_heads,
                                                   activation=None if i == self.n_layer - 1 else torch.relu,
                                                   num_heads=num_heads))
                elif self.gnn == 'sage':
                    agg = 'pool'
                    self.gnn_layers.append(SAGEConv(in_feats=feature_len if i == 0 else self.dim,
                                                    out_feats=self.dim,
                                                    activation=None if i == self.n_layer - 1 else torch.relu,
                                                    aggregator_type=agg))
                elif self.gnn == 'tag':
                    hops = 2
                    self.gnn_layers.append(TAGConv(in_feats=feature_len if i == 0 else self.dim,
                                                   out_feats=self.dim,
                                                   activation=None if i == self.n_layer - 1 else torch.relu,
                                                   k=hops))
        elif self.gnn == 'sgc':
            self.gnn_layers.append(SGConv(in_feats=feature_len, out_feats=self.dim, k=self.n_layer))
        else:
            raise ValueError('unknown GNN model')
        self.pooling_layer = SumPooling()
        self.factor = None

    def forward(self, graph):
        feature = graph.ndata['feature']
        h = one_hot(feature, num_classes=self.feature_len)
        h = torch.sum(h, dim=1, dtype=torch.float)
        for layer in self.gnn_layers:
            h = layer(graph, h)
            if self.gnn == 'gat':
                h = torch.reshape(h, [h.size()[0], -1])
        if self.factor is None:
            self.factor = math.sqrt(self.dim) / float(torch.mean(torch.linalg.norm(h, dim=1)))
        h *= self.factor
        graph_embedding = self.pooling_layer(graph, h)
        return graph_embedding


class RelationEmb(torch.nn.Module):
    def __init__(self, dim):
        super(RelationEmb, self).__init__()
        self.dim = dim
        self.para = nn.Parameter(torch.FloatTensor(14, self.dim))
        nn.init.xavier_uniform_(self.para)

    def forward(self, bond1, bond2):
        bond_diff = bond1.long() - bond2.long()
        relation_emb = torch.mm(bond_diff, self.para.long())
        return relation_emb


class CSGL(torch.nn.Module):
    def __init__(self, args, feature_len, molesule_dataloader, adj_graph, bond_feature):
        super(CSGL, self).__init__()
        
        self.args = args
        self.low_level_GNN = GNN(args, feature_len)
        self.high_level_on = args.high_level_on
        self.rel_emb = RelationEmb(args.dim)
        self.molesule_dataloader = molesule_dataloader
        self.adj_graph = adj_graph
        self.bond_feature = bond_feature  
        self.neighbor_size = args.neighbor_size

        if self.high_level_on == 1:
            if self.args.att == 1:
                self.att_linear = nn.Linear(3*self.args.dim, 1)
                nn.init.xavier_uniform_(self.att_linear.weight)
                
            if self.args.layer_agg == 'cat':
                self.cat_linear = nn.Linear(2*self.args.dim, self.args.dim)
                nn.init.xavier_uniform_(self.att_linear.weight)

        
    def get_low_level_emb(self):
        all_molecule_embeddings = []
        for graphs in self.molesule_dataloader:
            molecule_embeddings = self.low_level_GNN(graphs)
            all_molecule_embeddings.append(molecule_embeddings)
        self.all_molecule_embeddings = torch.cat(all_molecule_embeddings, dim=0)

        return self.all_molecule_embeddings

    def get_high_level_emb(self, dataloader):
        
        all_reactant_embeddings = []
        all_product_embeddings = []

        for i, batch in enumerate(dataloader):
            r_id, p_id = batch[:, 0].long(), batch[:, 1].long()
            r_bond = batch[:, 2:16].long()
            p_bond = batch[:, 16:30].long()

            reactant_embeddings, product_embeddings = self.high_level_GNN(r_id, p_id, r_bond, p_bond)
            all_reactant_embeddings.append(reactant_embeddings)
            all_product_embeddings.append(product_embeddings)

        all_reactant_embeddings = torch.cat(all_reactant_embeddings, dim=0)
        all_product_embeddings = torch.cat(all_product_embeddings, dim=0)

        return all_reactant_embeddings, all_product_embeddings


    def high_level_GNN(self, r_id, p_id, r_bond, p_bond):
        r_id = r_id.long()
        p_id = p_id.long()

        all_molecule_embeddings = self.get_low_level_emb()

        r_emb = all_molecule_embeddings[r_id]
        p_emb = all_molecule_embeddings[p_id]

        if self.high_level_on == 1:
            r_neighbor = torch.index_select(self.adj_graph, 0, r_id)  
            p_neighbor = torch.index_select(self.adj_graph, 0, p_id)  
            
            r_neighbor_emb = all_molecule_embeddings[r_neighbor.long()] 
            p_neighbor_emb = all_molecule_embeddings[p_neighbor.long()] 
            
            r_bond_repeated = r_bond.unsqueeze(1).repeat(1, self.neighbor_size, 1).view(-1, 14)
            p_bond_repeated = p_bond.unsqueeze(1).repeat(1, self.neighbor_size, 1).view(-1, 14)

            r_neighbor_bond = self.bond_feature[r_neighbor.long()].view(-1, 14) 
            p_neighbor_bond = self.bond_feature[p_neighbor.long()].view(-1, 14) 

            r_neighbor_rel = self.rel_emb(r_bond_repeated, r_neighbor_bond).view(-1, self.neighbor_size, self.args.dim)
            p_neighbor_rel = self.rel_emb(p_bond_repeated, p_neighbor_bond).view(-1, self.neighbor_size, self.args.dim)
            
            if self.args.att == 1:
                r_nei_agg = self.att_nei_emb(r_emb, r_neighbor_emb, r_neighbor_rel)
                p_nei_agg = self.att_nei_emb(p_emb, p_neighbor_emb, p_neighbor_rel)
            else:
                r_nei_agg = (1.0/self.neighbor_size) * torch.sum(r_neighbor_emb, 1)
                p_nei_agg = (1.0/self.neighbor_size) * torch.sum(p_neighbor_emb, 1)
            
            r_emb_1 = r_emb + r_nei_agg
            p_emb_1 = p_emb + p_nei_agg

            if self.args.layer_agg == 'sum':
                final_r_emb = r_emb + r_emb_1
                final_p_emb = p_emb + p_emb_1
            elif self.args.layer_agg == 'cat':
                final_r_emb = self.cat_linear(torch.cat([r_emb, r_emb_1], dim=1))
                final_p_emb = self.cat_linear(torch.cat([p_emb, p_emb_1], dim=1))
            elif self.args.layer_agg == 'last':
                final_r_emb = r_emb_1
                final_p_emb = p_emb_1

        elif self.high_level_on == 0:
            final_r_emb = r_emb
            final_p_emb = p_emb

        else:
            raise ValueError('unknown model')

        return final_r_emb, final_p_emb

    def att_nei_emb(self, source_emb, nei_emb, nei_rel):
        source_emb = source_emb.unsqueeze(1).repeat(1, self.neighbor_size, 1)  
        
        cat_vec = torch.cat((source_emb, nei_emb, nei_rel), dim=1).view(-1, 3*self.args.dim)
        score_att = self.att_linear(cat_vec).view(-1, self.neighbor_size, 1)
        score_att = F.softmax(score_att.float(), 1)
        nei_emb_feature = torch.sum(score_att * nei_emb, 1)

        return nei_emb_feature

    def forward(self, r_id, p_id, r_bond, p_bond):

        r_emb, p_emb = self.high_level_GNN(r_id, p_id, r_bond, p_bond)

        return r_emb, p_emb
       



class DecoderModel(torch.nn.Module):
    def __init__(self, args):
        super(DecoderModel, self).__init__()
        
        self.dim = args.dim
        self.pre_decoder = args.pre_decoder
        self.dataset = args.dataset
        
        if self.pre_decoder == 'LR':
            if self.dataset == 'USPTO-MTL':
                self.pre = nn.Linear(2 * self.dim, 1000)
            else:
                self.pre = nn.Linear(2 * self.dim, 46)

        elif self.pre_decoder == 'MLP':
            if self.dataset == 'USPTO-MTL':
                self.pre1 = nn.Linear(2 * self.dim, 512)
                self.pre2 = nn.Linear(512, 1000)
            else:
                self.pre1 = nn.Linear(2 * self.dim, 256)
                self.pre2 = nn.Linear(256, 46)
            self.leakyrelu = nn.LeakyReLU()

            nn.init.xavier_uniform_(self.pre1.weight)
            nn.init.xavier_uniform_(self.pre2.weight)
        
        else:
            raise ValueError('unknown pre_decoder model')
        

    def forward(self, input_vec):

        pre_vec = input_vec.float()

        if self.pre_decoder == 'LR':
            pre_result = self.pre(pre_vec)
        elif self.pre_decoder == 'MLP':
            pre_result = self.pre2(self.leakyrelu(self.pre1(pre_vec)))
        else:
            raise ValueError('unknown pre_decoder model')

        return pre_result 
        