import os
import torch
import pickle
import data_processing
import numpy as np
from model import CSGL
from copy import deepcopy
from dgl.dataloading import GraphDataLoader

from torch.utils.data import DataLoader, Dataset
import torch.utils.data as Data


def train(args, data):
    feature_encoder, molesule_graphs, train_data, test_data, valid_data, adj_graph, bond_feature = data
    feature_len = sum([len(feature_encoder[key]) for key in data_processing.attribute_names])
    
    molesule_dataloader = GraphDataLoader(molesule_graphs, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    model = CSGL(args, feature_len, molesule_dataloader, adj_graph, bond_feature)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_delay)


    if torch.cuda.is_available():
        model = model.cuda(args.gpu)

    best_model_params = None
    best_val_mrr = 0



    model.eval()
    log_eval, val_mrr = evaluate(model, 'valid', valid_data, args)
    log_test, _ = evaluate(model, 'test', test_data, args)

    for epoch in range(args.epoch):


        rxn_dataloader = Data.DataLoader(dataset=train_data, batch_size = args.batch_size, shuffle=True, drop_last=True)
        
        model.train()
        for i, batch in enumerate(rxn_dataloader):
            r_id, p_id = batch[:, 0].long(), batch[:, 1].long()

            r_bond = batch[:, 2:16].long()
            p_bond = batch[:, 16:30].long()


            reactant_embeddings, product_embeddings = model(r_id, p_id, r_bond, p_bond)

            loss = calculate_loss(reactant_embeddings, product_embeddings, r_bond, p_bond, model, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        log_eval, val_mrr = evaluate(model, 'valid', valid_data, args)
        log_test, _ = evaluate(model, 'test', test_data, args)

        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_model_params = deepcopy(model.state_dict())


    model.load_state_dict(best_model_params)
    log_test, _ = evaluate(model, 'test', test_data, args)


    if args.save_model:
        if not os.path.exists('../saved/'):
            os.mkdir('../saved/')

        directory = '../saved/%s_%s_%s_%d_%d' % (args.task, args.dataset, args.gnn, args.dim, args.high_level_on)

        if not os.path.exists(directory):
            os.mkdir(directory)

        torch.save(best_model_params, directory + '/model.pt')
        with open(directory + '/hparams.pkl', 'wb') as f:
            hp_dict = {'gnn': args.gnn, 'layer': args.layer, 'feature_len': feature_len, 'dim': args.dim}
            pickle.dump(hp_dict, f)
        with open(directory + '/feature_enc.pkl', 'wb') as f:
            pickle.dump(feature_encoder, f)


def calculate_loss(reactant_embeddings, product_embeddings, r_bond, p_bond, model, args):

    
    head_embedding = reactant_embeddings
    tail_embedding = product_embeddings
    head_bond = r_bond
    tail_bond = p_bond

    mini_batch_size = 32
    distance_matrix = torch.zeros(args.batch_size, args.batch_size)

    for i in range(0, args.batch_size, mini_batch_size):
        for j in range(0, args.batch_size, mini_batch_size):
            i_end = min(i + mini_batch_size, args.batch_size)
            j_end = min(j + mini_batch_size, args.batch_size)
            
            head_bond_batch = head_bond[i:i_end].unsqueeze(1).expand(-1, j_end-j, -1).reshape(-1, 14)
            tail_bond_batch = tail_bond[j:j_end].unsqueeze(0).expand(i_end-i, -1, -1).reshape(-1, 14)

            relation_emb_batch = model.rel_emb(head_bond_batch, tail_bond_batch).view(i_end-i, j_end-j, -1)

            head_emb_expanded = head_embedding[i:i_end].unsqueeze(1).expand(-1, j_end-j, -1)
            tail_emb_expanded = tail_embedding[j:j_end].unsqueeze(0).expand(i_end-i, -1, -1)

            distances_batch = torch.norm(head_emb_expanded + relation_emb_batch - tail_emb_expanded, dim=2)

            distance_matrix[i:i_end, j:j_end] = distances_batch

    
    dist = distance_matrix
    pos = torch.diag(dist)
    mask = torch.eye(args.batch_size)
    if torch.cuda.is_available():
        mask = mask.cuda(args.gpu)
    neg = (1 - mask) * dist + mask * args.margin
    neg = torch.relu(args.margin - neg)
    loss = torch.mean(pos) + torch.sum(neg) / args.batch_size / (args.batch_size - 1)

    return loss



def evaluate(model, mode, data, args):
    model.eval()
    with torch.no_grad():
        
        rxn_dataloader = Data.DataLoader(dataset=data, batch_size = args.batch_size)
        all_reactant_embeddings, all_product_embeddings = model.get_high_level_emb(rxn_dataloader)

        all_rankings = []
        rxn_dataloader1 = Data.DataLoader(dataset=data, batch_size = 1)
        i = 0
        for i, batch in enumerate(rxn_dataloader1):
            
            r_id, p_id = batch[:, 0].long(), batch[:, 1].long()
            r_bond, p_bond = batch[:, 2:16].long(), batch[:, 16:30].long()
            r_emb = all_reactant_embeddings[i]
            
            repeat_r_emb = r_emb.repeat(len(all_product_embeddings), 1)
            repeat_r_bond = r_bond.repeat(len(all_product_embeddings), 1)
            repeat_p_bond = p_bond.repeat(len(all_product_embeddings), 1)

            relation_vec = model.rel_emb(repeat_r_bond, repeat_p_bond) 

            dist = torch.sqrt( torch.sum((repeat_r_emb + relation_vec-all_product_embeddings)**2, dim=1) )
            rank_list = torch.argsort(dist, dim=0)
            rank_num = np.where(rank_list == i)[0][0] + 1
            all_rankings.append(rank_num)
            i = i + 1

        all_rankings = np.array(all_rankings)
        mrr = float(np.mean(1 / all_rankings))
        mr = float(np.mean(all_rankings))
        h1 = float(np.mean(all_rankings <= 1))
        h3 = float(np.mean(all_rankings <= 3))
        h5 = float(np.mean(all_rankings <= 5))
        h10 = float(np.mean(all_rankings <= 10))

        loggg = '%s  mrr: %.4f  mr: %.4f  h1: %.4f  h3: %.4f  h5: %.4f  h10: %.4f' % (mode, mrr, mr, h1, h3, h5, h10)

        return loggg, mrr
