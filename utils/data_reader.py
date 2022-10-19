from utils import config
import numpy as np
import os
import pickle
import re
from collections import defaultdict


class Lang:
    """
    create a new word dictionary, including 3 dictionaries:
    1) word to index;
    2) word and its count;
    3) index to word;
    and one counter indicating the number of words.
    """

    def __init__(self, init_index2word):
        """
        :param init_index2word: a dictionary containing (id: token) pairs
        """
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)  # Count default tokens

    def index_words(self, sentence):
        for word in sentence:
            if re.search(r'\d', word):
                continue
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def read_langs(vocab):
    with open('data/vocabulary.txt', 'r') as f:
        lines = f.readlines()
    for word in lines:
        vocab.index_word(word.strip())

    data_train, data_dev, data_test = defaultdict(list), defaultdict(list), defaultdict(list)
    data_dict = config.data_dict
    train_dialog = np.load(data_dict + '/sys_dialog_texts.train.npy', allow_pickle=True)
    train_target = np.load(data_dict + '/sys_target_texts.train.npy', allow_pickle=True)
    train_emotion = np.load(data_dict + '/sys_emotion_texts.train.npy', allow_pickle=True)

    data_train['dialog'] = train_dialog
    data_train['target'] = train_target
    data_train['emotion'] = train_emotion
    assert len(data_train['dialog']) == len(data_train['target']) == len(data_train['emotion'])

    dev_dialog = np.load(data_dict + '/sys_dialog_texts.dev.npy', allow_pickle=True)
    dev_target = np.load(data_dict + '/sys_target_texts.dev.npy', allow_pickle=True)
    dev_emotion = np.load(data_dict + '/sys_emotion_texts.dev.npy', allow_pickle=True)

    data_dev['dialog'] = dev_dialog
    data_dev['target'] = dev_target
    data_dev['emotion'] = dev_emotion
    assert len(data_dev['dialog']) == len(data_dev['target']) == len(data_dev['emotion'])

    test_dialog = np.load(data_dict + '/sys_dialog_texts.test.npy', allow_pickle=True)
    test_target = np.load(data_dict + '/sys_target_texts.test.npy', allow_pickle=True)
    test_emotion = np.load(data_dict + '/sys_emotion_texts.test.npy', allow_pickle=True)

    data_test['dialog'] = test_dialog
    data_test['target'] = test_target
    data_test['emotion'] = test_emotion
    assert len(data_test['dialog']) == len(data_test['target']) == len(data_test['emotion'])

    return data_train, data_dev, data_test, vocab


def read_graph(vocab):
    data_train, data_dev, data_test = defaultdict(list), defaultdict(list), defaultdict(list)
    data_dict = config.data_dict
    with open('data/ConceptVocabulary.txt', 'r') as f:
        lines = f.readlines()
    vocab = set(vocab) & {word.strip() for word in lines}

    graph_vocab = Lang({idx: w for idx, w in enumerate(vocab)})

    train_graph_GAE = np.load(data_dict + '/sys_AE_graphs.train.npy', allow_pickle=True)
    dev_graph_GAE = np.load(data_dict + '/sys_AE_graphs.dev.npy', allow_pickle=True)
    test_graph_GAE = np.load(data_dict + '/sys_AE_graphs.test.npy', allow_pickle=True)

    for graph in train_graph_GAE:
        nodes, edges, masks = graph['nodes'][:800], graph['edges'], graph['masks'][:800]
        node2index = {word: idx for idx, word in enumerate(nodes) if word in vocab}
        index2mask = {node2index[w]: m for (w, m) in zip(nodes, masks)}
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
        data_train["nodes"].append([x[0] for x in sorted(node2index.items(), key=lambda item: item[1])])
        data_train["value_post"].append(value_post)
        data_train["value_prior"].append(value_prior)
        data_train["row"].append(row)
        data_train["col"].append(col)

    for graph in dev_graph_GAE:
        nodes, edges, masks = graph['nodes'][:800], graph['edges'], graph['masks'][:800]
        node2index = {word: idx for idx, word in enumerate(nodes) if word in vocab}
        index2mask = {node2index[w]: m for (w, m) in zip(nodes, masks)}
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
        data_dev["nodes"].append([x[0] for x in sorted(node2index.items(), key=lambda item: item[1])])
        data_dev["value_post"].append(value_post)
        data_dev["value_prior"].append(value_prior)
        data_dev["row"].append(row)
        data_dev["col"].append(col)

    for graph in test_graph_GAE:
        nodes, edges, masks = graph['nodes'][:800], graph['edges'], graph['masks'][:800]
        node2index = {word: idx for idx, word in enumerate(nodes) if word in vocab}
        index2mask = {node2index[w]: m for (w, m) in zip(nodes, masks)}
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
        data_test["nodes"].append([x[0] for x in sorted(node2index.items(), key=lambda item: item[1])])
        data_test["value_post"].append(value_post)
        data_test["value_prior"].append(value_prior)
        data_test["row"].append(row)
        data_test["col"].append(col)

    return data_train, data_dev, data_test, graph_vocab


def load_dataset():
    if os.path.exists(config.data_path):
        print("LOADING empathetic_dialogue")
        with open(config.data_path, "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building dataset...")
        data_tra, data_val, data_tst, vocab = read_langs(vocab=Lang(
            {config.PAD_idx: "PAD", config.EOS_idx: "EOS", config.SOS_idx: "SOS", config.UNK_idx: "UNK",
             config.USR_idx: "USR", config.SYS_idx: "SYS", config.CLS_idx: "CLS"}))
        with open(config.data_path, "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")

    if os.path.exists(config.graph_path):
        print("LOADING causal graphs")
        with open(config.graph_path, "rb") as f:
            [graph_tra, graph_val, graph_tst, graph_vocab] = pickle.load(f)
    else:
        print("Building causal graphs")
        graph_tra, graph_val, graph_tst, graph_vocab = read_graph(vocab.word2index)
        with open(config.graph_path, "wb") as f:
            pickle.dump([graph_tra, graph_val, graph_tst, graph_vocab], f)
            print("Save PICKLE")

    data_tra.update(graph_tra)
    data_val.update(graph_val)
    data_tst.update(graph_tst)

    for i in range(3):
        print('[emotion]:', data_tra['emotion'][i])
        print('[context]:', [' '.join(u) for u in data_tra['dialog'][i]])
        print('[target]:', ' '.join(data_tra['target'][i]))
        print(" ")

    return data_tra, data_val, data_tst, vocab, graph_vocab
