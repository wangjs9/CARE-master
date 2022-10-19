""" Manage beam search info structure.
    Heavily borrowed from OpenNMT-py.
    For code in OpenNMT-py, please check the following link:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""

import torch
import torch.nn.functional as F
from utils import config


class Beam():
    ''' Beam search '''

    def __init__(self, size, device=False):

        self.size = size
        self._done = False

        # The score for each translation on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((size,), config.PAD_idx, dtype=torch.long, device=device)]
        self.next_ys[0][0] = config.SOS_idx

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        "Update beam status and check if finished or not."
        num_words = word_prob.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)  # 2nd sort

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = torch.floor_divide(best_scores_id, num_words)
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0].item() == config.EOS_idx:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[config.SOS_idx] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))


class Translator(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, model, lang):

        self.model = model
        self.lang = lang
        self.vocab_size = lang.n_words
        self.beam_size = config.beam_size
        self.device = config.device

    def beam_search(self, batch, max_dec_step):
        ''' Translation work in one batch '''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(batch, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(batch, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)

            active_encoder_db = None

            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_encoder_db, active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(inst_dec_beams, len_dec_seq, enc_output, inst_idx_to_position_map, n_bm, mask_src):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def predict_word(dec_seq, enc_output, n_active_inst, n_bm, mask_src_tuple):
                ## masking
                mask_trg = dec_seq.data.eq(config.PAD_idx).unsqueeze(1)
                mask_seq, mask_graph = mask_src_tuple
                mask_seq = torch.cat([mask_seq[0].unsqueeze(0)] * mask_trg.size(0), 0)
                mask_graph = torch.cat([mask_graph[0].unsqueeze(0)] * mask_trg.size(0), 0)

                out, attn_dist = self.model.decoder(self.model.embedding(dec_seq), enc_output,
                                                    ((mask_seq, mask_graph), mask_trg))
                prob = self.model.generator(out)

                word_prob = prob[:, -1]
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]
                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            word_prob = predict_word(dec_seq, enc_output, n_active_inst, n_bm, mask_src)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            # -- Encode
            batch = {k: t.to(config.device) if type(t) != list else t for k, t in batch.items()}
            enc_batch = batch["input_batch"]
            nodes = [n.to(self.device) for n in batch["nodes"]]
            node_num = batch["node_num"]
            adj_prior = [adj.to(self.device) for adj in batch["adj_prior"]]

            # encode input sequence
            mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
            src_enc = self.model.embedding(enc_batch) + self.model.embedding(batch["mask_input"])
            src_enc = self.model.encoder(src_enc, mask_src)

            # compute the latent variable
            random_variable = torch.randn_like(src_enc[:, 0:1, :])
            ctx_hidden, _ = self.model.hidden_mapping(random_variable, src_enc, src_enc, mask_src)
            latent_seq = self.model.SeqVAE.generate(ctx_hidden)

            # encode causal graph based on the condition including context hidden, sequence latent, emotion
            ns_embed = self.model.node_embedding(torch.stack(nodes, 0))
            emotion_emb = self.model.emotion_embedding(batch["emotion_label"])
            conds = torch.cat((ctx_hidden, emotion_emb.unsqueeze(1), latent_seq.unsqueeze(1)), dim=1)  #
            conds = F.leaky_relu(self.model.condition_mapping(conds))
            relations, relation_masks = [], []
            for i, (ns, adj) in enumerate(zip(nodes, adj_prior)):
                rel, mask = self.model.GVAE.generate(ns_embed[i][:node_num[i]], adj, conds[i:i + 1, :])
                relations.append(rel)
                relation_masks.append(mask)
            rel_enc = torch.stack(relations, dim=0)
            mask_rel = torch.stack(relation_masks, dim=0).unsqueeze(1)

            # -- Repeat data for beam search
            n_bm = self.beam_size
            n_inst, len_s = enc_batch.size()
            d_h = src_enc.size(-1)
            batch = enc_batch.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, -1, d_h)
            rel_enc = rel_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, -1, d_h)

            # -- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)]

            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            # -- Decode
            for len_dec_seq in range(1, max_dec_step + 1):

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams,
                    len_dec_seq,
                    (src_enc, rel_enc),
                    inst_idx_to_position_map,
                    n_bm,
                    (mask_src, mask_rel)
                )

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                batch, encoder_db, src_enc, inst_idx_to_position_map = collate_active_info(
                    batch,
                    src_enc,
                    inst_idx_to_position_map,
                    active_inst_idx_list
                )

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)

        ret_sentences = []
        for d in batch_hyp:
            ret_sentences.append(' '.join([self.model.vocab.index2word[idx] for idx in d[0]]).replace('EOS', ''))

        return ret_sentences  # , batch_scores


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = seq_range_expand
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand
