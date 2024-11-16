import os
import argparse
import data_processing
import train





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the index of gpu device')

    parser.add_argument('--task', type=str, default='pretrain_pp', help='downstream task')
    parser.add_argument('--dataset', type=str, default='USPTO', help='dataset name')
    parser.add_argument('--epoch', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--gnn', type=str, default='gcn', help='name of the GNN model')
    parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
    parser.add_argument('--dim', type=int, default=256, help='dimension of molecule embeddings')
    parser.add_argument('--neighbor_size', type=int, default=2, help='neighbor sample size')
    parser.add_argument('--att', type=int, default=1, help='attention')
    parser.add_argument('--layer_agg', type=str, default='sum', help='sum, cat, last')
    parser.add_argument('--margin', type=float, default=4.0, help='margin in contrastive loss')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--ratio', type=float, default=1.0, help='training ratio')
    parser.add_argument('--save_model', type=bool, default=True, help='save the trained model to disk')
    parser.add_argument('--high_level_on', type=int, default=1, help='if high_level or not')
    parser.add_argument('--weight_delay', type=float, default=1e-5, help='regularization')


    
    args = parser.parse_args()


    data = data_processing.load_data(args)
    train.train(args, data)


if __name__ == '__main__':
    main()
