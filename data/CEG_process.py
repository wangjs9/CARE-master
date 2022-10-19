from tqdm import tqdm
from collections import defaultdict
import json
import os
import numpy as np
import random
from itertools import product


def CausalGraph():
    cause_effect_dic = {}
    with open('data/Cause_Effect_Graph/Cause_Effect_Graph.txt', 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    for line in tqdm(lines, total=len(lines)):
        head, tail, scores = line.split('\t')
        if head == tail:
            continue
        score = sum([float(s) for s in scores.split('\\')])
        if score < 0.1:
            continue
        cause_effect_dic[(head, tail)] = score
    new_cause_effect_dic = {}
    for (head, tail), score in tqdm(cause_effect_dic.items(), total=len(cause_effect_dic)):
        if (tail, head) in new_cause_effect_dic or (head, tail) in new_cause_effect_dic:
            continue
        elif (tail, head) in cause_effect_dic:
            re_score = cause_effect_dic[(tail, head)]
            if score > re_score:
                new_cause_effect_dic[(head, tail)] = score
            else:
                new_cause_effect_dic[(tail, head)] = re_score
        else:
            new_cause_effect_dic[(head, tail)] = score
    del cause_effect_dic
    assert len([(h, t) for (h, t) in new_cause_effect_dic.keys() if (t, h) in new_cause_effect_dic.keys()]) == 0
    cause_dic, effect_dic = defaultdict(list), defaultdict(list)
    for (head, tail), score in tqdm(new_cause_effect_dic.items(), total=len(new_cause_effect_dic)):
        cause_dic[tail].append((head, score))
        effect_dic[head].append((tail, score))
    del new_cause_effect_dic
    cause_dic = {w: [k[0] for k in sorted(lst, key=lambda x: x[1], reverse=True)] for w, lst in cause_dic.items()}
    effect_dic = {w: [k[0] for k in sorted(lst, key=lambda x: x[1], reverse=True)] for w, lst in effect_dic.items()}
    with open('data/Cause_Effect_Graph/cause_dic.json', 'w') as f:
        json.dump(cause_dic, f)
    with open('data/Cause_Effect_Graph/effect_dic.json', 'w') as f:
        json.dump(effect_dic, f)


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


def one_causal_graph(nodes, default_k=1600):
    global CauseDict, EffectDict, Vocabulary, AllEmotion
    forbid_words = set()
    for w in nodes:
        forbid_words.update(CauseDict.get(w, {}))
        forbid_words.update(EffectDict.get(w, {}))
    forbid_words = forbid_words - set(nodes)
    if len(forbid_words) < default_k - len(nodes):
        sel_num = len(forbid_words)
    else:
        sel_num = default_k - len(nodes)
    other_words = random.sample(forbid_words, sel_num)
    edges = []
    masks = [1] * len(nodes) + [0] * len(other_words)
    for head, tail in product(nodes, nodes):
        if head == tail:
            continue
        if head in CauseDict.get(tail, []):
            edges.append([head, tail])

    nodes.extend(other_words)
    return {'nodes': nodes, 'edges': edges, 'masks': masks, 'emotion': nodes[0]}
