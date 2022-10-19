import torch
import torch.utils.data as data
import logging
from utils import config
import pprint
import scipy.sparse as sp
import numpy as np

pp = pprint.PrettyPrinter(indent=1)
from utils.data_reader import load_dataset
from torch.utils.data import RandomSampler

torch.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class Dataset(data.Dataset):
    def __init__(self, data, vocab, graph_vocab=None):
        self.vocab = vocab
        if graph_vocab is None:
            self.graph_vocab = vocab
        else:
            self.graph_vocab = graph_vocab
        self.data = data
        self.emo_map = {
            'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6,
            'lonely': 7, 'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12,
            'hopeful': 13, 'anxious': 14, 'disappointed': 15, 'joyful': 16, 'prepared': 17, 'guilty': 18,
            'furious': 19, 'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23,
            'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29,
            'apprehensive': 30, 'faithful': 31
        }

    def __len__(self):
        return len(self.data['target'])

    def __getitem__(self, index):
        item = {}
        item["dialog_text"] = self.data["dialog"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]

        item["dialog"], item["dialog_mask"] = self.preprocess(item["dialog_text"])
        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["emotion_label"] = self.emo_map[item["emotion_text"]]

        nodes = [self.graph_vocab.word2index[word] for word in self.data["nodes"][index]]
        item["node_num"] = len(nodes)
        value_post, value_prior = self.data["value_post"][index], self.data["value_prior"][index]
        row, col = self.data["row"][index], self.data["col"][index]
        adj_post = sp.csr_matrix((value_post, (row, col)), shape=(len(nodes), len(nodes)))
        adj_prior = sp.csr_matrix((value_prior, (row, col)), shape=(len(nodes), len(nodes)))
        adj_post.eliminate_zeros()
        adj_prior.eliminate_zeros()
        # Some preprocessing
        adj_norm_post = preprocess_graph(adj_post)
        adj_norm_prior = preprocess_graph(adj_prior)
        adj_label = adj_post + sp.eye(adj_post.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())
        nodes.extend([0] * (1600 - len(nodes)))
        item["nodes"] = torch.LongTensor(nodes)
        item["adj_post"] = adj_norm_post.to_dense()
        item["adj_prior"] = adj_norm_prior.to_dense()
        item["adj_label"] = adj_label

        return item

    def preprocess(self, arr, anw=False):
        """Converts words to ids."""
        if (anw):
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                        arr] + [config.EOS_idx]
            return torch.LongTensor(sequence)
        else:
            X_dial = [config.CLS_idx]
            X_mask = [config.CLS_idx]
            for i, sentence in enumerate(arr):
                X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                           sentence]
                spk = self.vocab.word2index["USR"] if i % 2 == 0 else self.vocab.word2index["SYS"]
                X_mask += [spk for _ in range(len(sentence))]
            assert len(X_dial) == len(X_mask)
            if len(X_dial) > 1024:
                X_dial = X_dial[-1024:]
                X_mask = X_mask[-1024:]
            return torch.LongTensor(X_dial), torch.LongTensor(X_mask)


def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["dialog"]), reverse=True)  ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    d = {}
    # sequence
    input_batch, input_lengths = merge(item_info['dialog'])
    mask_input, mask_input_lengths = merge(item_info['dialog_mask'])
    d["input_batch"] = input_batch
    d["input_lengths"] = torch.LongTensor(input_lengths)
    d["mask_input"] = mask_input
    target_batch, target_lengths = merge(item_info['target'])
    d["target_batch"] = target_batch
    d["target_lengths"] = torch.LongTensor(target_lengths)
    d["input_txt"] = item_info['dialog_text']
    d["target_txt"] = item_info['target_text']
    # emotiom
    d["emotion_label"] = torch.LongTensor(item_info['emotion_label'])
    d["emotion_txt"] = item_info['emotion_text']

    # graph
    d["nodes"] = item_info["nodes"]
    d["node_num"] = torch.LongTensor(item_info['node_num'])
    d["adj_post"] = item_info["adj_post"]
    d["adj_prior"] = item_info["adj_prior"]
    d["adj_label"] = item_info["adj_label"]

    return d


def prepare_data_seq(batch_size=32):
    torch.multiprocessing.set_start_method('spawn')
    pairs_tra, pairs_val, pairs_tst, vocab, _ = load_dataset()
    logging.info("Vocab  {} ".format(vocab.n_words))
    dataset_train = Dataset(pairs_tra, vocab, )

    dataset_valid = Dataset(pairs_val, vocab)
    dev_sampler = RandomSampler(dataset_valid, replacement=False)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_valid,
        num_workers=4,
        # cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l
        # cat /proc/cpuinfo| grep "cpu cores"| uniq
        batch_size=batch_size,
        sampler=dev_sampler,
        collate_fn=collate_fn,
        pin_memory=False
    )

    dataset_test = Dataset(pairs_tst, vocab)
    data_loader_tst = torch.utils.data.DataLoader(
        dataset=dataset_test,
        num_workers=8,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=False
    )

    return dataset_train, data_loader_val, data_loader_tst, vocab
