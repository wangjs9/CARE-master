import os
import pickle
import torch
import torch.utils.data as data
import logging
from utils.data_loader import preprocess_graph
import pprint
import scipy.sparse as sp
import numpy as np
from collections import defaultdict

pp = pprint.PrettyPrinter(indent=1)
from utils.data_reader import Lang
from torch.utils.data import RandomSampler

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

graph_path = './data_graph/graphs_lists.p'
causal_graph_path = './data_graph/causal_graph_{}.npy'


def read_langs(vocab, num=5):
    data_train, data_dev = defaultdict(list), defaultdict(list)
    for i in range(1, num + 1):
        graph_GVE = np.load(causal_graph_path.format(str(i)), allow_pickle=True)
        for graph in graph_GVE:
            nodes, edges, masks, emotion = graph['nodes'], graph['edges'], graph['masks'], graph['emotion']
            node2index = {word: idx for idx, word in enumerate(nodes) if word in vocab}
            index2mask = {node2index[w]: m for (w, m) in zip(nodes, masks) if w in node2index}
            row, col = [], []
            value_prior = []
            for (h, t) in edges:
                hid = node2index.get(h, -1)
                tid = node2index.get(t, -1)
                if tid > 0 and hid > 0:
                    row.extend([hid, tid])
                    col.extend([tid, hid])
                    if index2mask[tid] * index2mask[hid] > 0:
                        value_prior.extend([1] * 2)
                    else:
                        value_prior.extend([0] * 2)
            value_post = [1] * len(row)
            data_train["nodes"].append(sorted(node2index.keys(), key=lambda item: item[1]))
            data_train["value_post"].append(value_post)
            data_train["row"].append(row)
            data_train["col"].append(col)
            data_train["emotion"].append(emotion)

    for i in range(num + 1, num + 2):
        graph_GVE = np.load(causal_graph_path.format(str(i)), allow_pickle=True)
        for graph in graph_GVE:
            nodes, edges, masks, emotion = graph['nodes'], graph['edges'], graph['masks'], graph['emotion']
            node2index = {word: idx for idx, word in enumerate(nodes) if word in vocab}
            index2mask = {node2index[w]: m for (w, m) in zip(nodes, masks) if w in node2index}
            row, col = [], []
            value_prior = []
            for (h, t) in edges:
                hid = node2index.get(h, -1)
                tid = node2index.get(t, -1)
                if tid > 0 and hid > 0:
                    row.extend([hid, tid])
                    col.extend([tid, hid])
                    if index2mask[tid] * index2mask[hid] > 0:
                        value_prior.extend([1] * 2)
                    else:
                        value_prior.extend([0] * 2)
            value_post = [1] * len(row)
            data_dev["nodes"].append(sorted(node2index.keys(), key=lambda item: item[1]))
            data_dev["value_post"].append(value_post)
            data_dev["row"].append(row)
            data_dev["col"].append(col)
            data_dev["emotion"].append(emotion)

    return data_train, data_dev


def load_graph_dataset():
    if os.path.exists(graph_path):
        print('LOADING graph data')
        with open(graph_path, "rb") as f:
            [data_tra, data_val, graph_vocab] = pickle.load(f)
    else:
        print('Building dataset...')
        with open('./data/vocabulary.txt', 'r') as f:
            vocab = f.readlines()
            vocab = {v.strip() for v in vocab}
        with open('./data/ConceptVocabulary.txt', 'r') as f:
            lines = f.readlines()
        vocab = set(vocab) & {word.strip() for word in lines}
        graph_vocab = Lang({idx: w for idx, w in enumerate(vocab)})
        data_tra, data_val = read_langs(graph_vocab.word2index)
        with open(graph_path, "wb") as f:
            pickle.dump([data_tra, data_val, graph_vocab], f)
            print("Save PICKLE")

    return data_tra, data_val, graph_vocab


class Graph_Dataset(data.Dataset):
    def __init__(self, graph_data, graph_vocab):
        self.graph_vocab = graph_vocab
        self.emo_map = {
            'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6,
            'lonely': 7, 'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12,
            'hopeful': 13, 'anxious': 14, 'disappointed': 15, 'joyful': 16, 'prepared': 17, 'guilty': 18,
            'furious': 19, 'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23,
            'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29,
            'apprehensive': 30, 'faithful': 31
        }
        self.graph_data = graph_data

    def __len__(self):
        return len(self.graph_data['emotion'])

    def __getitem__(self, index):
        item = {}
        item["emotion_label"] = self.emo_map[self.graph_data["emotion"][index]]
        nodes = [self.graph_vocab.word2index[word] for word in self.graph_data["nodes"][index]]
        value_post = self.graph_data["value_post"][index]
        row, col = self.graph_data["row"][index], self.graph_data["col"][index]
        adj_post = sp.csr_matrix((value_post, (row, col)), shape=(len(nodes), len(nodes)))
        adj_post.eliminate_zeros()

        # Some preprocessing
        adj_norm_post = preprocess_graph(adj_post)
        adj_label = adj_post + sp.eye(adj_post.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())
        item["nodes"] = torch.LongTensor(nodes)
        item["adj_post"] = adj_norm_post.to_dense()
        item["adj_label"] = adj_label

        return item


def collate_fn(graph_data):
    graph_data.sort(key=lambda x: len(x["nodes"]), reverse=True)  ## sort by source seq
    item_info = {}
    for key in graph_data[0].keys():
        item_info[key] = [d[key] for d in graph_data]

    d = {}
    d["emotion_label"] = torch.LongTensor(item_info['emotion_label'])
    d["nodes"] = item_info["nodes"]
    d["adj_post"] = item_info["adj_post"]
    d["adj_label"] = item_info["adj_label"]

    return d


def prepare_data_graph(batch_size=32):
    torch.multiprocessing.set_start_method('spawn')
    data_tra, data_val, graph_vocab = load_graph_dataset()
    logging.info("Vocab  {} ".format(graph_vocab.n_words))
    dataset_train = Graph_Dataset(data_tra, graph_vocab)

    dataset_valid = Graph_Dataset(data_val, graph_vocab)
    dev_sampler = RandomSampler(dataset_valid, replacement=False)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_valid,
        num_workers=8,
        batch_size=batch_size,
        sampler=dev_sampler,
        collate_fn=collate_fn,
        pin_memory=False
    )

    return dataset_train, data_loader_val, graph_vocab
