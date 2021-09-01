import os
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn

# path to a pretrained word embedding file
word_emb_path = '/home/henry/glove/glove.840B.300d.txt'
assert(word_emb_path is not None)


username = Path.home().name
project_dir = Path(__file__).resolve().parent.parent
sdk_dir = project_dir.joinpath('CMU-MultimodalSDK')
data_dir = project_dir.joinpath('datasets')
data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath(
    'MOSEI'), 'ur_funny': data_dir.joinpath('UR_FUNNY')}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
}

criterion_dict = {
    'mosi': 'L1Loss',
    'iemocap': 'CrossEntropyLoss',
    'ur_funny': 'CrossEntropyLoss'
}

def get_args():
    parser = argparse.ArgumentParser(description='MOSI-and-MOSEI Sentiment Analysis')
    parser.add_argument('-f', default='', type=str)

    # Tasks
    parser.add_argument('--dataset', type=str, default='mosi', choices=['mosi','mosei'],
                        help='dataset to use (default: mosei)')
    parser.add_argument('--data_path', type=str, default='datasets',
                        help='path for storing the dataset')

    # Dropouts
    parser.add_argument('--dropout_a', type=float, default=0.1,
                        help='dropout of acoustic LSTM out layer')
    parser.add_argument('--dropout_v', type=float, default=0.1,
                        help='dropout of visual LSTM out layer')
    parser.add_argument('--dropout_prj', type=float, default=0.1,
                        help='dropout of projection layer')

    # Architecture
    parser.add_argument('--multiseed', action='store_true', help='training using multiple seed')
    parser.add_argument('--contrast', action='store_true', help='using contrast learning')
    parser.add_argument('--add_va', action='store_true', help='if add va MMILB module')
    parser.add_argument('--n_layer', type=int, default=1,
                        help='number of layers in LSTM encoders (default: 1)')
    parser.add_argument('--cpc_layers', type=int, default=1,
                        help='number of layers in CPC NCE estimator (default: 1)')
    parser.add_argument('--d_vh', type=int, default=16,
                        help='hidden size in visual rnn')
    parser.add_argument('--d_ah', type=int, default=16,
                        help='hidden size in acoustic rnn')
    parser.add_argument('--d_vout', type=int, default=16,
                        help='output size in visual rnn')
    parser.add_argument('--d_aout', type=int, default=16,
                        help='output size in acoustic rnn')
    parser.add_argument('--bidirectional', action='store_true', help='Whether to use bidirectional rnn')
    parser.add_argument('--d_prjh', type=int, default=128,
                        help='hidden size in projection network')
    parser.add_argument('--pretrain_emb', type=int, default=768,
                        help='dimension of pretrained model output')


    # Activations
    parser.add_argument('--mmilb_mid_activation', type=str, default='ReLU',
                        help='Activation layer type in the middle of all MMILB modules')
    parser.add_argument('--mmilb_last_activation', type=str, default='Tanh',
                        help='Activation layer type at the end of all MMILB modules')
    parser.add_argument('--cpc_activation', type=str, default='Tanh',
                        help='Activation layer type in all CPC modules')

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size (default: 32)')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clip value (default: 0.8)')
    parser.add_argument('--lr_main', type=float, default=1e-3,
                        help='initial learning rate for main model parameters (default: 1e-3)')
    parser.add_argument('--lr_bert', type=float, default=5e-5,
                        help='initial learning rate for bert parameters (default: 5e-5)')
    parser.add_argument('--lr_mmilb', type=float, default=1e-3,
                        help='initial learning rate for mmilb parameters (default: 1e-3)')
    parser.add_argument('--alpha', type=float, default=0.1, help='weight for CPC NCE estimation item (default: 0.1)')
    parser.add_argument('--beta', type=float, default=0.1, help='weight for lld item (default: 0.1)')

    parser.add_argument('--weight_decay_main', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_bert', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_club', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
        
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='number of epochs (default: 40)')
    parser.add_argument('--when', type=int, default=20,
                        help='when to decay learning rate (default: 20)')
    parser.add_argument('--patience', type=int, default=10,
                        help='when to stop training if best never change')
    parser.add_argument('--update_batch', type=int, default=1,
                        help='update batch interval')

    # Logistics
    parser.add_argument('--log_interval', type=int, default=100,
                        help='frequency of result logging (default: 100)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    args = parser.parse_args()
    return args


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, data, mode='train'):
        """Configuration Class: set kwargs as class attributes with setattr"""
        self.dataset_dir = data_dict[data.lower()]
        self.sdk_dir = sdk_dir
        self.mode = mode
        # Glove path
        self.word_emb_path = word_emb_path

        # Data Split ex) 'train', 'valid', 'test'
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(dataset='mosi', mode='train', batch_size=32):
    config = Config(data=dataset, mode=mode)
    
    config.dataset = dataset
    config.batch_size = batch_size

    return config