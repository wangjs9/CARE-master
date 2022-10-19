from collections import defaultdict
import re
import os
import json
import math
import numpy as np
import nltk
from itertools import product
import spacy
from tqdm import tqdm
from multiprocess.pool import Pool
import sys
from CEG_process import CausalGraph

nltk.download('punkt')
nlp = spacy.load('en_core_web_md', disable=['ner', 'parser', 'textcat'])
useless_word = {'hear', 'happen', 'happened', 'happening', 'hope', 'glad', 'true', 'okey', 'bad', 'wow',
                'good', 'better', 'feel', 'sound', 'fun', 'nice', 'sad', 'great', 'time', 'happy', 'felt',
                'yeah', 'terrible', 'hard', 'awesome'}


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


def clean(sentence, word_pairs):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)
    return sentence


def clean_data():
    vocab = Lang({})
    data_train, data_dev, data_test = defaultdict(list), defaultdict(list), defaultdict(list)
    word_pairs = {"it's": "it is", "don't": "do not", "doesn't": "does not", "didn't": "did not", "you'd": "you would",
                  "you're": "you are", "you'll": "you will", "i'm": "i am", "they're": "they are", "that's": "that is",
                  "what's": "what is", "couldn't": "could not", "i've": "i have", "we've": "we have", "can't": "cannot",
                  "i'd": "i would", "i'd": "i would", "aren't": "are not", "isn't": "is not", "wasn't": "was not",
                  "weren't": "were not", "won't": "will not", "there's": "there is", "there're": "there are"}
    train_context = np.load('data/empathetic-dialogue/sys_dialog_texts.train.npy', allow_pickle=True)
    train_target = np.load('data/empathetic-dialogue/sys_target_texts.train.npy', allow_pickle=True)
    train_situation = np.load('data/empathetic-dialogue/sys_situation_texts.train.npy', allow_pickle=True)

    dev_context = np.load('data/empathetic-dialogue/sys_dialog_texts.dev.npy', allow_pickle=True)
    dev_target = np.load('data/empathetic-dialogue/sys_target_texts.dev.npy', allow_pickle=True)
    dev_situation = np.load('data/empathetic-dialogue/sys_situation_texts.dev.npy', allow_pickle=True)

    test_context = np.load('data/empathetic-dialogue/sys_dialog_texts.test.npy', allow_pickle=True)
    test_target = np.load('data/empathetic-dialogue/sys_target_texts.test.npy', allow_pickle=True)
    test_situation = np.load('data/empathetic-dialogue/sys_situation_texts.test.npy', allow_pickle=True)

    for context in train_context:
        u_list = []
        for u in context:
            u = clean(u, word_pairs)
            u_list.append(u)
            vocab.index_words(u)
        data_train['context'].append(u_list)
    np.save('data/sys_dialog_texts.train.npy', data_train['context'])
    for target in train_target:
        target = clean(target, word_pairs)
        data_train['target'].append(target)
        vocab.index_words(target)
    np.save('data/sys_target_texts.train.npy', data_train['target'])
    for situation in train_situation:
        situation = clean(situation, word_pairs)
        data_train['situation'].append(situation)
        vocab.index_words(situation)
    np.save('data/sys_situation_texts.train.npy', data_train['situation'])
    past_c, past_id = '', -1
    for idx, c in enumerate(train_situation):
        if c != past_c:
            past_c = c
            past_id = idx
        data_train['index'].append(past_id)
    np.save('data/sys_index_texts.train.npy', data_train['index'])

    for context in dev_context:
        u_list = []
        for u in context:
            u = clean(u, word_pairs)
            u_list.append(u)
            vocab.index_words(u)
        data_dev['context'].append(u_list)
    np.save('data/sys_dialog_texts.dev.npy', data_dev['context'])
    for target in dev_target:
        target = clean(target, word_pairs)
        data_dev['target'].append(target)
        vocab.index_words(target)
    np.save('data/sys_target_texts.dev.npy', data_dev['target'])
    for situation in dev_situation:
        situation = clean(situation, word_pairs)
        data_dev['situation'].append(situation)
        vocab.index_words(situation)
    np.save('data/sys_situation_texts.dev.npy', data_dev['situation'])
    past_c, past_id = '', -1
    for idx, c in enumerate(dev_situation):
        if c != past_c:
            past_c = c
            past_id = idx
        data_dev['index'].append(past_id)
    np.save('data/sys_index_texts.dev.npy', data_dev['index'])
    with open('data/vocabulary.txt', 'w') as f:
        for idx, word in vocab.index2word.items():
            f.write(word)
            f.write('\n')

    for context in test_context:
        u_list = []
        for u in context:
            u = clean(u, word_pairs)
            u_list.append(u)
        data_test['context'].append(u_list)
    np.save('data/sys_dialog_texts.test.npy', data_test['context'])
    for target in test_target:
        target = clean(target, word_pairs)
        data_test['target'].append(target)
    np.save('data/sys_target_texts.test.npy', data_test['target'])
    for situation in test_situation:
        situation = clean(situation, word_pairs)
        data_test['situation'].append(situation)
    np.save('data/sys_situation_texts.test.npy', data_test['situation'])
    past_c, past_id = '', -1
    for idx, c in enumerate(test_situation):
        if c != past_c:
            past_c = c
            past_id = idx
        data_test['index'].append(past_id)
    np.save('data/sys_index_texts.test.npy', data_test['index'])


def hard_ground(sent):
    """
    extract concpets from sent
    :param sent: a list of words
    :return: keywords
    """
    global Vocabulary
    doc = nlp(' '.join(list(set(sent))))
    res = set()
    for t in doc:
        if t.text in Vocabulary and t.text not in res:
            res.add(t.text)
        elif t.lemma_ in Vocabulary and t.lemma_ not in res:
            res.add(t.lemma_)
        elif t.pos == 'VERB':
            word = t.text + 'ed'
            if nlp(word)[0].pos_ != 'VERB':
                continue
            if word in Vocabulary and word not in res:
                res.add(word)
    return res


def Graph_Construct(future=True):
    """
    we make the graph to a sequence, where nodes appear according to the order in the dialogue
    :return:
    """
    global CauseDict, EffectDict
    CauseDict = json.load(open('data/Cause_Effect_Graph/cause_dic.json', 'r'))
    EffectDict = json.load(open('data/Cause_Effect_Graph/effect_dic.json', 'r'))
    CauseDict['proud'] = CauseDict['pride']
    EffectDict['proud'] = EffectDict['pride']

    global Vocabulary
    if not os.path.exists('data/ConceptVocabulary.txt'):
        keywords = set(CauseDict.keys()) | set(EffectDict.keys())
        with open('data/ConceptVocabulary.txt', 'w') as f:
            for k in keywords:
                f.write(k)
                f.write('\n')
    with open('data/ConceptVocabulary.txt', 'r') as f:
        lines = f.readlines()
    Vocabulary = {word.strip() for word in lines}
    with open('data/vocabulary.txt', 'r') as f:
        lines = f.readlines()
        lines = {word.strip() for word in lines}
    Vocabulary = Vocabulary & lines

    for tn in ['train', 'dev', 'test']:
        target = np.load('data/sys_target_texts.{}.npy'.format(tn), allow_pickle=True)
        dialogue = np.load('data/sys_dialog_texts.{}.npy'.format(tn), allow_pickle=True)
        emotion = np.load('data/sys_emotion_texts.{}.npy'.format(tn), allow_pickle=True)
        index = np.load('data/sys_index_texts.{}.npy'.format(tn), allow_pickle=True)

        current_graphs, next_graphs = [], []
        for i, idx in tqdm(enumerate(index), total=len(target)):
            keywords = hard_ground(np.hstack(dialogue[i][::2]))
            graph = one_graph_construct(keywords, str(emotion[i]))
            current_graphs.append(graph)
            if i != idx:
                if future:
                    res_keywords = hard_ground(target[i - 1]) - useless_word
                    graph = one_graph_construct(keywords | res_keywords, str(emotion[i]))
                next_graphs.append(graph)
            if len(target) == i + 1 or i + 1 == index[i + 1]:
                if future:
                    res_keywords = hard_ground(target[i]) - useless_word
                    graph = one_graph_construct(keywords | res_keywords, str(emotion[i]))
                next_graphs.append(graph)

        assert len(current_graphs) == len(next_graphs)
        np.save('data/sys_current_graphs.{}.npy'.format(tn), current_graphs)
        np.save('data/sys_next_graphs.{}.npy'.format(tn), next_graphs)


def one_graph_construct(keywords, emotion):
    if len(keywords) == 0:
        return {'nodes': [emotion], 'heads': [], 'tails': []}
    else:
        heads, tails = [], []
        global CauseDict, EffectDict
        keywords = [k for k in keywords if k != emotion]
        for head, tail in product(keywords, keywords):
            if head == tail:
                continue
            if head in CauseDict.get(tail, []):
                heads.append(head)
                tails.append(tail)
        for word in keywords:
            if word in CauseDict[emotion]:
                heads.append(word)
                tails.append(emotion)
            elif word in EffectDict[emotion]:
                heads.append(emotion)
                tails.append(word)
        keywords.insert(0, emotion)
        return {'nodes': keywords, 'heads': heads, 'tails': tails}


def Graph_Extraction():
    """
    we extract a subgraph from the causal graph
    :return:
    """
    global CauseDict, EffectDict
    CauseDict = json.load(open('data/Cause_Effect_Graph/cause_dic.json', 'r'))
    EffectDict = json.load(open('data/Cause_Effect_Graph/effect_dic.json', 'r'))
    CauseDict['proud'] = CauseDict['pride']
    EffectDict['proud'] = EffectDict['pride']

    global Vocabulary
    with open('data/ConceptVocabulary.txt', 'r') as f:
        lines = f.readlines()
    Vocabulary = {word.strip() for word in lines}
    with open('data/vocabulary.txt', 'r') as f:
        lines = f.readlines()
        lines = {word.strip() for word in lines}
    Vocabulary = Vocabulary & lines

    for tn in ['train', 'dev', 'test']:
        current_graphs = np.load('data/sys_current_graphs.{}.npy'.format(tn), allow_pickle=True)
        next_graphs = np.load('data/sys_next_graphs.{}.npy'.format(tn), allow_pickle=True)
        dialog = np.load('data/sys_dialog_texts.{}.npy'.format(tn), allow_pickle=True)

        graphs_AE = []
        _num = 20
        _num_len = math.ceil(len(current_graphs) / _num)
        data_list = [[current_graphs[i:i + _num_len], next_graphs[i:i + _num_len], dialog[i:i + _num_len], tn == 'test']
                     for i in range(0, len(current_graphs), _num_len)]
        pool = Pool(processes=_num)
        graph_list = pool.map_async(pool_graph_extract, data_list)
        pool.close()
        pool.join()
        for graphs in graph_list.get():
            graphs_AE.extend(graphs)

        np.save('data/sys_AE_graphs.{}.npy'.format(tn), graphs_AE)


def pool_graph_extract(data):
    known_graph_list, unknown_graph_list, query_list, test = data
    graph_list = []
    for (known_graph, unknown_graph, query) in tqdm(zip(known_graph_list, unknown_graph_list, query_list),
                                                    total=len(known_graph_list)):
        graph = one_graph_extract(known_graph, unknown_graph, ' '.join(query[-1]), test)
        # graph = one_graph_extract(known_graph, unknown_graph, ' '.join([' '.join(l) for l in query[::2]]), test)
        graph_list.append(graph)
    return graph_list


def one_graph_extract(known_graph, unknown_graph, query, test=False):
    global CauseDict, EffectDict, Vocabulary
    known_words, unknown_words = known_graph['nodes'], unknown_graph['nodes']
    possible_head, possible_tail = unknown_graph['heads'], unknown_graph['tails']
    nodes = [x for x in known_words]
    masks = [1] * len(nodes)
    extend_words = set()

    for w in known_words:
        extend_words.update(set(CauseDict.get(w, [])[:300]))
        extend_words.update(set(EffectDict.get(w, [])[:300]))
    sent_nlp, emotion_nlp = nlp(query), nlp(known_words[0])
    extend_words = extend_words & Vocabulary - set(known_words)
    word_score = {w: sent_nlp.similarity(nlp(w)) for w in extend_words}
    if not test:
        for w in set(unknown_words) - set(known_words):
            word_score[w] = 2
    extract_words = sorted(word_score.items(), key=lambda item: item[1], reverse=True)[:(1600 - len(known_words))]
    extract_words = [x[0] for x in extract_words]
    nodes.extend(list(extract_words))
    masks.extend([0] * len(extract_words))
    if not test:
        unknown_words_nlp = {w: nlp(w) for w in unknown_words}
        replace_word_dic = defaultdict(list)
        for ew in extract_words:
            if ew in unknown_words_nlp:
                continue
            else:
                ew_nlp = nlp(ew)
                for uw, uw_nlp in unknown_words_nlp.items():
                    if ew_nlp.similarity(uw_nlp) > 0.75:
                        replace_word_dic[uw].append(ew)
        edges = []
        for (h, t) in zip(possible_head, possible_tail):
            edges.append((h, t))
            if h in replace_word_dic:
                for r in replace_word_dic[h]:
                    edges.append((r, t))
            if t in replace_word_dic:
                for r in replace_word_dic[t]:
                    edges.append((h, r))
    else:
        edges = [(h, t) for (h, t) in zip(possible_head, possible_tail)]

    return {'nodes': nodes, 'edges': edges, 'masks': masks}
    # return len(extract_words), len([n for n in unknown_words if n in extract_words])/ (len(unknown_words) - len(known_words) + 1e-5)


if __name__ == '__main__':
    clean_data()
    if not os.path.exists('data/Cause_Effect_Graph/cause_dic.json') and \
            not os.path.exists('data/Cause_Effect_Graph/effect_dic.json'):
        CausalGraph()
        print('Cause Effect Dictionary Finished.')
    Graph_Construct()
    Graph_Extraction()
