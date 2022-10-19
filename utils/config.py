import argparse, os
import torch

parser = argparse.ArgumentParser()
"""hyper parameters"""
parser.add_argument('--embed_dim', type=int, default=300)
parser.add_argument('--hidden_dim', type=int, default=300)
parser.add_argument('--trs_num_layer', type=int, default=2)
parser.add_argument('--trs_num_head', type=int, default=4)
parser.add_argument('--trs_depth', type=int, default=40)
parser.add_argument('--trs_filter', type=int, default=50)
parser.add_argument('--bz', type=int, default=16, help='the size of batch')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay')
parser.add_argument('--noam', type=bool, default=True)
parser.add_argument('--label_smoothing', type=bool, default=True)
parser.add_argument('--schedule', type=int, default=10000)
parser.add_argument("--act", action="store_true")
parser.add_argument('--test', action='store_true')
parser.add_argument('--bow', action='store_true')
parser.add_argument('--kl_step', type=int, default=1600)
parser.add_argument('--max_k', type=int, default=-1)
parser.add_argument('--min_step', type=int, default=10000)
parser.add_argument('--beam_size', type=int, default=5)
parser.add_argument('--save_path', type=str, default='')
parser.add_argument('--recover_path', type=str, default='')
parser.add_argument('--seed', type=int, default=0)


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    cmd = 'python main.py '
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
            if opts.__dict__[key]:
                cmd += '--{}={} '.format(key, opts.__dict__[key])
    print('=' * 80)
    if not opts.__dict__['test']:
        save_path = 'save/{}/setting.txt'.format(opts.__dict__['save_path']) if opts.__dict__[
            'save_path'] else 'save/CARE/setting.txt'
        with open(save_path, 'w') as f:
            f.write(cmd)


arg = parser.parse_args()
"""datasat and paths for CausalBank, conceptNet and dataset"""

data_dict = 'data/'
data_path = 'data/dataset_preproc.p'
# graph_path = 'data/graph_preproc.p' if not arg.small_graph else 'data/graph_preproc_800.p'
graph_path = 'data/graph_preproc.p'
if arg.save_path == '':
    save_path = 'save/CARE'
else:
    save_path = 'save/{}/'.format(arg.save_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(save_path + 'test/')
recover_path = arg.recover_path
embed_path = 'data/embedding.txt'
node_embed_path = 'data/graph_embedding.txt'
glove_embed = 'data/glove.6B/glove.6B.300d.txt'
"""hyper parameters"""
embed_dim = arg.embed_dim
hidden_dim = arg.hidden_dim
trs_num_layer = arg.trs_num_layer
trs_num_head = arg.trs_num_head
trs_depth = arg.trs_depth
trs_filter = arg.trs_filter
num_emotion = 32
act = arg.act
beam_size = arg.beam_size
bow = arg.bow
topk = 5
seed = arg.seed
"""training parameters"""
test = arg.test
bz = arg.bz
lr = arg.lr
weight_decay = arg.weight_decay
schedule = arg.schedule
label_smoothing = arg.label_smoothing
noam = arg.noam
kl_step = arg.kl_step
max_k = arg.max_k
min_step = arg.min_step
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

UNK_idx, PAD_idx, EOS_idx, SOS_idx = 0, 1, 2, 3
USR_idx, SYS_idx, CLS_idx = 4, 5, 6
