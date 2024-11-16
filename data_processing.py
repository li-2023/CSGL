import os
import dgl
import torch
import pickle
import pysmiles
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

attribute_names = ['element', 'charge', 'aromatic', 'hcount']

class SmilesDataset(dgl.data.DGLDataset):
    def __init__(self, args, feature_encoder=None, raw_graphs=None):
        self.args = args
        self.feature_encoder = feature_encoder
        self.raw_graphs = raw_graphs
        
        self.path = '../data/' + self.args.dataset + '/cache/'
        self.graphs = []

        super().__init__(name='Smiles')

    def to_gpu(self):
        if torch.cuda.is_available():
            self.graphs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.graphs]

    def save(self):
        dgl.save_graphs(self.path + 'graphs.bin', self.graphs)

    def load(self):
        self.graphs = dgl.load_graphs(self.path + 'graphs.bin')[0]

        self.to_gpu()

    def process(self):

        for i, raw_graph in enumerate(self.raw_graphs):
            graph = networkx_to_dgl(raw_graph, self.feature_encoder)
            self.graphs.append(graph)

        self.to_gpu()

    def has_cache(self):
        return os.path.exists(self.path + 'graphs.bin')

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)


def networkx_to_dgl(raw_graph, feature_encoder):
    
    src = [s for (s, _) in raw_graph.edges]
    dst = [t for (_, t) in raw_graph.edges]
    graph = dgl.graph((src, dst), num_nodes=len(raw_graph.nodes))
    node_features = []
    for i in range(len(raw_graph.nodes)):
        raw_feature = raw_graph.nodes[i]
        numerical_feature = []
        for j in attribute_names:

            if raw_feature[j] in feature_encoder[j]:

                numerical_feature.append(feature_encoder[j][raw_feature[j]])
            else:
                numerical_feature.append(feature_encoder[j]['unknown'])
        node_features.append(numerical_feature)
    node_features = torch.tensor(node_features)
    graph.ndata['feature'] = node_features

    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)
    return graph


def read_data(dataset):
    path = '../data/' + dataset + '/file_id.txt'


    all_values = defaultdict(set)
    graphs = []

    with open(path) as f:
        for line in f.readlines():
            idx, smiles = line.strip().split('\t')



            if '[se]' in smiles:
                smiles = smiles.replace('[se]', '[Se]')

            molecule_graph = pysmiles.read_smiles(smiles, zero_order_bonds=False)

            for attr in attribute_names:
                for _, value in molecule_graph.nodes(data=attr):
                    all_values[attr].add(value)

            graphs.append(molecule_graph)
    
        return all_values, graphs



def get_feature_encoder(all_values):
    feature_encoder = {}
    idx = 0
    
    for key, values in all_values.items():
        feature_encoder[key] = {}
        for value in values:
            feature_encoder[key][value] = idx
            idx += 1
        feature_encoder[key]['unknown'] = idx
        idx += 1

    return feature_encoder


def preprocess(dataset):


    all_values, graphs = read_data(dataset)

    feature_encoder = get_feature_encoder(all_values)

    
    path = '../data/' + dataset + '/cache/feature_encoder.pkl'

    with open(path, 'wb') as f:
        pickle.dump(feature_encoder, f)

    return feature_encoder, graphs



def func_statistic(input_smiles_str):

    smiles_list = input_smiles_str.split(".")


    total_bonds = 0
    total_rings = 0
    
    bond_types = {
        "C-C": 0, "C=C": 0, "C#C": 0, "C-O": 0, "C=O": 0, "C-N": 0, "C=N": 0, 
        "C#N": 0, "C-S": 0, "C=P": 0, "C-Si": 0, 'C-X': 0  
    }


    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        total_bonds += mol.GetNumBonds()
        
        sssr = Chem.GetSymmSSSR(mol)
        total_rings += len(sssr)  
        
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            a1_symbol = a1.GetSymbol()
            a2_symbol = a2.GetSymbol()
            bond_order = bond.GetBondTypeAsDouble() 

            if a1_symbol in ['Cl', 'Br', 'I', 'F'] or a2_symbol in ['Cl', 'Br', 'I', 'F']:
                bond_key = 'C-X'
            else:
                bond_key = f"{a1_symbol}-{a2_symbol}" if a1_symbol <= a2_symbol else f"{a2_symbol}-{a1_symbol}"
                if bond_order == 1.0:
                    bond_key = f"{bond_key}"
                elif bond_order == 2.0:
                    bond_key = f"{bond_key}="
                elif bond_order == 3.0:
                    bond_key = f"{bond_key}#"

            if bond_key in bond_types:
                bond_types[bond_key] += 1

    list_bond = [int(total_bonds), int(total_rings)]

    for bond_type, count in bond_types.items():
        list_bond.append(int(count))

    return list_bond



def load_rxn(args):

    if os.path.exists('../data/' + args.dataset + '/rxn_cache/'):


        path = '../data/' + args.dataset + '/rxn_cache/molecular_num_dict.pkl'
        with open(path, 'rb') as f:
            molecular_num_dict = pickle.load(f)
        
        molecular_bond_feature = np.load("../data/" + args.dataset + "/rxn_cache/molecular_bond_feature.npy")

        train_data = np.load("../data/" + args.dataset + "/rxn_cache/train.npy")
        test_data  = np.load("../data/" + args.dataset + "/rxn_cache/test.npy")
        valid_data = np.load("../data/" + args.dataset + "/rxn_cache/valid.npy")
        train_reactant_bonds = np.load("../data/" + args.dataset + "/rxn_cache/train_reactant_bonds.npy")
        train_product_bonds  = np.load("../data/" + args.dataset + "/rxn_cache/train_product_bonds.npy")
        test_reactant_bonds  = np.load("../data/" + args.dataset + "/rxn_cache/test_reactant_bonds.npy")
        test_product_bonds   = np.load("../data/" + args.dataset + "/rxn_cache/test_product_bonds.npy")
        valid_reactant_bonds = np.load("../data/" + args.dataset + "/rxn_cache/valid_reactant_bonds.npy")
        valid_product_bonds  = np.load("../data/" + args.dataset + "/rxn_cache/valid_product_bonds.npy")

    else:

        path = '../data/' + args.dataset + '/rxn_cache/'
        os.mkdir(path)
        
        id_file = "../data/" + args.dataset + "/file_id.txt"
        molecular_num_dict = dict()
        molecular_bond_feature = []

        with open(id_file) as f:
            for line in f.readlines():
                id_num, smiles = line.strip().split('\t')
                if smiles not in molecular_num_dict:
                    molecular_num_dict[smiles] = []
                    molecular_num_dict[smiles].append(id_num)

                    bond_feature = func_statistic(smiles)

                    molecular_num_dict[smiles].append(bond_feature)

                    molecular_bond_feature.append(bond_feature)

        molecular_bond_feature = np.array(molecular_bond_feature)


        train_file = "../data/" + args.dataset + "/train.txt"
        train_list = []
        train_reactant_bonds = []
        train_product_bonds = []
        with open(train_file) as f:
            for line in f.readlines():
                idx, reactant, product = line.strip().split('\t')
                train_list.append([int(molecular_num_dict[reactant][0]), int(molecular_num_dict[product][0])])
                train_reactant_bonds.append(molecular_num_dict[reactant][1])
                train_product_bonds.append(molecular_num_dict[product][1])
        train_data = np.array(train_list)
        train_reactant_bonds = np.array(train_reactant_bonds)
        train_product_bonds = np.array(train_product_bonds)


        valid_file = "../data/" + args.dataset + "/valid.txt"
        valid_list = []
        valid_reactant_bonds = []
        valid_product_bonds = []
        with open(valid_file) as f:
            for line in f.readlines():
                idx, reactant, product = line.strip().split('\t')
                valid_list.append([int(molecular_num_dict[reactant][0]), int(molecular_num_dict[product][0])])
                valid_reactant_bonds.append(molecular_num_dict[reactant][1])
                valid_product_bonds.append(molecular_num_dict[product][1])
        valid_data = np.array(valid_list)
        valid_reactant_bonds = np.array(valid_reactant_bonds)
        valid_product_bonds = np.array(valid_product_bonds)



        test_file = "../data/" + args.dataset + "/test.txt"
        test_list = []
        test_reactant_bonds = []
        test_product_bonds = []
        with open(test_file) as f:
            for line in f.readlines():
                idx, reactant, product = line.strip().split('\t')
                test_list.append([int(molecular_num_dict[reactant][0]), int(molecular_num_dict[product][0])])
                test_reactant_bonds.append(molecular_num_dict[reactant][1])
                test_product_bonds.append(molecular_num_dict[product][1])
        test_data = np.array(test_list)
        test_reactant_bonds = np.array(test_reactant_bonds)
        test_product_bonds = np.array(test_product_bonds)


        path = '../data/' + args.dataset + '/rxn_cache/molecular_num_dict.pkl'
        with open(path, 'wb') as f:
            pickle.dump(molecular_num_dict, f)

        np.save("../data/" + args.dataset + "/rxn_cache/molecular_bond_feature.npy", molecular_bond_feature)

        np.save("../data/" + args.dataset + "/rxn_cache/train.npy", train_data)
        np.save("../data/" + args.dataset + "/rxn_cache/test.npy", test_data)
        np.save("../data/" + args.dataset + "/rxn_cache/valid.npy", valid_data)
        np.save("../data/" + args.dataset + "/rxn_cache/train_reactant_bonds.npy", train_reactant_bonds)
        np.save("../data/" + args.dataset + "/rxn_cache/train_product_bonds.npy", train_product_bonds)
        np.save("../data/" + args.dataset + "/rxn_cache/test_reactant_bonds.npy", test_reactant_bonds)
        np.save("../data/" + args.dataset + "/rxn_cache/test_product_bonds.npy", test_product_bonds)
        np.save("../data/" + args.dataset + "/rxn_cache/valid_reactant_bonds.npy", valid_reactant_bonds)
        np.save("../data/" + args.dataset + "/rxn_cache/valid_product_bonds.npy", valid_product_bonds)



    if args.ratio < 1:
        train_indices = np.random.choice(list( range(len(train_data)) ), size=int(len(train_data) * args.ratio), replace=False)
        train_data = train_data[train_indices]
        train_reactant_bonds = train_reactant_bonds[train_indices]
        train_product_bonds = train_product_bonds[train_indices]
    
    if os.path.exists('../data/' + args.dataset + '/rxn_cache/adj.npy'):
        adj_graph = np.load("../data/" + args.dataset + "/rxn_cache/adj.npy")
    else:
        adj_graph = construct_adj(train_data, args, len(molecular_num_dict))
        np.save('../data/' + args.dataset + '/rxn_cache/adj.npy', adj_graph)

    train_info = torch.from_numpy(np.concatenate((train_data, train_reactant_bonds, train_product_bonds), axis=1))
    test_info  = torch.from_numpy(np.concatenate((test_data, test_reactant_bonds, test_product_bonds), axis=1))
    valid_info = torch.from_numpy(np.concatenate((valid_data, valid_reactant_bonds, valid_product_bonds), axis=1))
    
    adj_graph = torch.from_numpy(adj_graph)
    molecular_bond_feature = torch.from_numpy(molecular_bond_feature)



    return train_info, test_info, valid_info, adj_graph, molecular_bond_feature


def construct_adj(graph_np, args, node_size):
    
    graph_dict = construct_graph(graph_np, node_size)
    
    neighbor_matrix = np.zeros([node_size, args.neighbor_size], dtype=np.int32)

    for node_id in range(node_size):

        neighbor_info = graph_dict[node_id]
        num_neighbor = neighbor_info[0]
        neighbor = neighbor_info[1]
        
        if num_neighbor >= args.neighbor_size:
            sampled_indices = np.random.choice(list(range(num_neighbor)), size=args.neighbor_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(num_neighbor)), size=args.neighbor_size, replace=True)
        
        neighbor_matrix[node_id] = np.array([neighbor[i] for i in sampled_indices])
    
    return neighbor_matrix


def construct_graph(graph_np, num_node):
    graph_dict = dict()
    for link in graph_np:
        source = link[0]
        target = link[1]
        if source not in graph_dict:
            graph_dict[source] = [0, []]
        graph_dict[source][0] += 1
        graph_dict[source][1].append(target)

        source = link[0]
        target = link[1]
        if target not in graph_dict:
            graph_dict[target] = [0, []]
        graph_dict[target][0] += 1
        graph_dict[target][1].append(source)

    for node_id in range(num_node):
        source = node_id
        target = node_id
        if source not in graph_dict:
            graph_dict[source] = [0, []]
        graph_dict[source][0] += 1
        graph_dict[source][1].append(target)

    return graph_dict



def load_data(args):
    if os.path.exists('../data/' + args.dataset + '/cache/'):
        path = '../data/' + args.dataset + '/cache/feature_encoder.pkl'
        with open(path, 'rb') as f:
            feature_encoder = pickle.load(f)
        dataset = SmilesDataset(args)

    else:
        path = '../data/' + args.dataset + '/cache/'
        os.mkdir(path)
        feature_encoder, graphs = preprocess(args.dataset)
        dataset = SmilesDataset(args, feature_encoder, graphs)
    
    train_info, test_info, valid_info, adj_graph, bond_feature = load_rxn(args)
    
    return feature_encoder, dataset, train_info, test_info, valid_info, adj_graph, bond_feature