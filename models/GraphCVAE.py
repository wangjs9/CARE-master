import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models._common_layer import share_embedding, LabelSmoothing, NoamOpt, LayerNorm, MultiHeadAttention
from models._common_module import Encoder, Graph_Infused_Decoder, Generator
from utils import config
import pprint

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
pp = pprint.PrettyPrinter(indent=1)


class PriorNet(nn.Module):
    def __init__(self, condition_size, latent_size, hidden_size=[256, 512, 256], dropout=0.4):
        super(PriorNet, self).__init__()
        self.project = nn.Linear(condition_size, condition_size)
        self.act = nn.LeakyReLU()
        self.norm_layer = LayerNorm(condition_size)
        mu_models, var_models = [], []
        for idx, (input, output) in enumerate(zip([condition_size] + hidden_size, hidden_size + [latent_size])):
            mu_models.extend([nn.Linear(input, output), nn.LeakyReLU()])
            var_models.extend([nn.Linear(input, output), nn.LeakyReLU()])
        self.mu_mlp = nn.Sequential(*mu_models)
        self.var_mlp = nn.Sequential(*var_models)
        self.mu_norm_layer = LayerNorm(latent_size)
        self.var_norm_layer = LayerNorm(latent_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, condition):
        condition = self.norm_layer(self.act(self.project(condition)))
        mu = self.dropout(self.mu_mlp(condition))
        log_var = self.dropout(self.var_mlp(condition))
        mu = self.mu_norm_layer(mu)
        log_var = self.var_norm_layer(log_var)
        return mu, log_var


class RecognizeNet(nn.Module):
    def __init__(self, target_size, latent_size, hidden_size=[256, 256], dropout=0.4):
        super(RecognizeNet, self).__init__()
        self.project = nn.Linear(target_size, target_size, bias=False)
        self.act = nn.LeakyReLU()
        self.norm_layer = LayerNorm(target_size)
        mu_models, var_models = [], []
        for idx, (input, output) in enumerate(zip([target_size] + hidden_size, hidden_size + [latent_size])):
            mu_models.extend([nn.Linear(input, output), nn.LeakyReLU()])
            var_models.extend([nn.Linear(input, output), nn.LeakyReLU()])
        self.mu_mlp = nn.Sequential(*mu_models)
        self.var_mlp = nn.Sequential(*var_models)
        self.mu_norm_layer = LayerNorm(latent_size)
        self.var_norm_layer = LayerNorm(latent_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target):
        target = self.norm_layer(self.act(self.project(target)))
        mu = self.dropout(self.mu_mlp(target))
        log_var = self.dropout(self.var_mlp(target))
        mu = self.mu_norm_layer(mu)
        log_var = self.var_norm_layer(log_var)
        return mu, log_var


class SeqVAE(nn.Module):
    def __init__(self, hidden_dim):
        super(SeqVAE, self).__init__()
        self.net_prior = PriorNet(hidden_dim, hidden_dim, hidden_size=[512])
        self.net_recog = RecognizeNet(hidden_dim, hidden_dim, hidden_size=[256])

    def forward(self, pre_hidden, post_hidden, train):
        mu_prior, logvar_prior = self.net_prior(pre_hidden.squeeze(1))
        mu_post, logvar_post = self.net_recog(post_hidden.squeeze(1))
        if train:
            latent = self.reparameterize(mu_post, logvar_post)
        else:
            latent = self.reparameterize(mu_prior, logvar_prior)
        loss = self.loss_function(mu_post, logvar_post, mu_prior, logvar_prior)
        return latent, loss

    def generate(self, hidden):
        mu, logvar = self.net_prior(hidden.squeeze(1))
        latent = self.reparameterize(mu, logvar)
        return latent

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def loss_function(self, post_mu, post_var, prior_mu, prior_var):
        kld = 0.5 * torch.sum((prior_mu - post_mu).pow(2) / prior_var.exp() + post_var.exp() / prior_var.exp() - 1
                              - (post_var - prior_var))
        if kld.isnan() or kld.isinf():
            print()
        assert not kld.isnan()
        assert not kld.isinf()
        return kld


### https://github.com/zfjsail/gae-pytorch

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0.5, act=nn.LeakyReLU()):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.support = nn.Linear(in_features, out_features, bias=False)

    def forward(self, input, adj):
        support = self.support(self.dropout(input))
        output = torch.spmm(adj, self.act(support))
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class LatentInference(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout=0.3):
        super(LatentInference, self).__init__()
        self.hidden = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=nn.LeakyReLU())
        self.condition_attn = MultiHeadAttention(hidden_dim1, config.trs_depth, config.trs_depth, hidden_dim1, 2)
        self.mu = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=nn.LeakyReLU())
        self.var = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=nn.LeakyReLU())

    def forward(self, ns_emb, adj, condition):
        hidden = self.hidden(ns_emb, adj)
        hidden, _ = self.condition_attn(hidden.unsqueeze(0), condition, condition, None)
        return self.mu(hidden.squeeze(0), adj), self.var(hidden.squeeze(0), adj)  # mu, log_var


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, z):
        z = self.dropout(z)
        adj = torch.mm(z, z.t())
        # normal = nn.LayerNorm(adj.size()).to(adj.device)
        # return self.act(normal(adj)), self.act(adj)
        return self.act(adj)


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout=0.3, train_causal=False):
        super(GCNModelVAE, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.train_causal = train_causal
        # post encoding
        self.encode_post = LatentInference(input_feat_dim, hidden_dim1, hidden_dim2, dropout)
        # prior encoding
        self.encode_prior = LatentInference(input_feat_dim, hidden_dim1, hidden_dim2, dropout)
        # decoding
        self.decode = InnerProductDecoder(dropout, act=nn.Sigmoid())
        # get relation embeddings
        self.node_mapping = nn.Linear(input_feat_dim, hidden_dim2, bias=False)
        self.act = nn.LeakyReLU()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, ns_emb, adj, adj_prior, condition, labels, train=True):
        ns_emb = self.dropout(ns_emb)
        mu, logvar = self.encode_post(ns_emb, adj, condition)
        mu_prior, logvar_prior = self.encode_prior(ns_emb, adj_prior, condition)
        if train:
            z = self.reparameterize(mu, logvar)
        else:
            z = self.reparameterize(mu_prior, logvar_prior)
        recover_adj = self.decode(self.dropout(z))
        relations, rel_mask = self.rel_represent(ns_emb, recover_adj.triu(diagonal=1))
        loss = self.loss_function(recover_adj, labels, (mu, mu_prior), (logvar, logvar_prior))
        return relations, rel_mask, loss

    def generate(self, ns_emb, adj, condition):
        mu, logvar = self.encode_prior(ns_emb, adj, condition)
        z = self.reparameterize(mu, logvar)
        recover_adj = self.decode(z)
        if config.max_k == -1:
            relations, rel_mask = self.rel_represent(ns_emb, recover_adj.triu(diagonal=1))
        else:
            relations, rel_mask = self.rel_represent(ns_emb, recover_adj.triu(diagonal=1), max_k=config.max_k)
        return relations, rel_mask

    def rel_represent(self, ns_emb, adj, max_k=512):
        device = ns_emb.device
        ns_num, emb_size = ns_emb.size()
        ns_emb = self.act(self.node_mapping(ns_emb))
        ns_repeat = ns_emb.unsqueeze(0).repeat(ns_num, 1, 1)
        ns_matrix = ns_repeat + ns_repeat.transpose(0, 1)
        ns_matrix = ns_matrix.flatten(0, 1)
        rel_num = (ns_num ** 2 - ns_num) / 2
        if rel_num < max_k:
            index = adj.flatten().topk(k=rel_num)[1]
            selected_rel = torch.zeros((max_k, emb_size), dtype=ns_emb.dtype, device=device)
            selected_rel[:rel_num, :] = ns_matrix.index_select(0, index)
        else:
            index = adj.flatten().topk(k=max_k)[1]
            selected_rel = ns_matrix.index_select(0, index)
        rel_mask = torch.arange(max_k) >= rel_num
        return selected_rel, rel_mask.to(device)

    def loss_function(self, preds, labels, mu, logvar):
        n_nodes = labels.size(0)
        pos_weight = float(n_nodes ** 2 - labels.sum() + n_nodes) / (labels.sum() - n_nodes + 0.01)
        norm = n_nodes ** 2 / float(n_nodes ** 2 - labels.sum() + n_nodes)
        recons_loss = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
        # if recons_loss.isnan() or recons_loss.isinf():
        #     print()
        assert not recons_loss.isnan()
        assert not recons_loss.isinf()

        if self.train_causal:
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kld = 0.5 / n_nodes * torch.mean(torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar, dim=1))
        else:
            post_mu, prior_mu = mu
            post_logvar, prior_logvar = logvar
            kld = 0.5 / n_nodes * torch.mean(
                torch.sum((prior_mu - post_mu).pow(2) / prior_logvar.exp()
                          + post_logvar.exp() / prior_logvar.exp() - 1 - (post_logvar - prior_logvar), dim=1))
        # if kld.isnan() or kld.isinf():
        #     print()
        assert not kld.isnan()
        assert not kld.isinf()

        return recons_loss, kld


class CAREModel(nn.Module):
    def __init__(
            self,
            vocab,
            embed_dim,
            hidden_dim,
            emotion_number,
            device=config.device,
            model_file_path=None,
            load_optim=True,
            graph_vocab=None,
            **unused,
    ):
        super(CAREModel, self).__init__()
        self.i_epoch, self.i_step = 0, 0
        self.current_loss = 500
        # self.attempt = 64 // config.bz
        self.attempt = 128 // config.bz  #
        self.device = device
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.hidden_dim = hidden_dim
        # model to be saved
        self.embedding = share_embedding(vocab, embed_dim)
        if graph_vocab is None:
            self.node_embedding = self.embedding
        else:
            self.node_embedding = share_embedding(graph_vocab, embed_dim)
        # encoder
        self.encoder = Encoder(
            embed_dim,
            hidden_dim,
            num_layers=config.trs_num_layer,
            num_heads=config.trs_num_head,
            total_key_depth=config.trs_depth,
            total_value_depth=config.trs_depth,
            filter_size=config.trs_filter,
            universal=True
        )
        # sequence CVAE & graph CVAE
        self.hidden_mapping = MultiHeadAttention(hidden_dim, config.trs_depth, config.trs_depth, hidden_dim, 2)
        self.SeqVAE = SeqVAE(hidden_dim)
        self.emotion_embedding = nn.Embedding(emotion_number, hidden_dim)
        # condition mapping
        self.condition_mapping = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.GVAE = GCNModelVAE(hidden_dim, hidden_dim * 2, hidden_dim)

        # decoder
        self.decoder = Graph_Infused_Decoder(
            embed_dim,
            hidden_dim,
            num_layers=config.trs_num_layer,
            num_heads=config.trs_num_head,
            total_key_depth=config.trs_depth,
            total_value_depth=config.trs_depth,
            filter_size=config.trs_filter
        )
        self.generator = Generator(hidden_dim, self.vocab_size)
        self.generator.proj.weight = self.embedding.lut.weight
        # model end
        if config.label_smoothing:
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
        else:
            self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx, reduction='sum')
        self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)
        self.criterion_emo = nn.NLLLoss()

        if config.noam:
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=0,
                betas=(0.9, 0.98),
                eps=1e-9
            )
            self.scheduler = NoamOpt(hidden_dim, 1, 1000, self.optimizer)


        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.05,
                patience=5,
                threshold_mode='rel',
                cooldown=0,
                min_lr=1e-5,
            )

        if os.path.exists(model_file_path):
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            print(state.keys())
            self.i_epoch = state['epoch']
            self.i_step = state['step']
            self.current_loss = state['current_loss']
            self.embedding.load_state_dict(state['embedding'])
            self.node_embedding.load_state_dict(state['node_embedding'])
            self.encoder.load_state_dict(state['encoder'])
            self.hidden_mapping.load_state_dict(state['hidden_mapping'])
            self.SeqVAE.load_state_dict(state['SeqVAE'])
            self.emotion_embedding.load_state_dict(state['emotion_embedding'])
            self.condition_mapping.load_state_dict(state['condition_mapping'])
            self.GVAE.load_state_dict(state['GVAE'])
            self.decoder.load_state_dict(state['decoder'])
            self.generator.load_state_dict(state['generator'])
            if load_optim:
                self.scheduler = pickle.loads(state['optimizer'])

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = os.path.join(self.model_dir, "best_model")

    def save_model(self):
        state = {
            'epoch': self.i_epoch,
            'step': self.i_step,
            'current_loss': self.current_loss,
            'embedding': self.embedding.state_dict(),
            'node_embedding': self.node_embedding.state_dict(),
            'encoder': self.encoder.state_dict(),
            'hidden_mapping': self.hidden_mapping.state_dict(),
            'SeqVAE': self.SeqVAE.state_dict(),
            'emotion_embedding': self.emotion_embedding.state_dict(),
            'condition_mapping': self.condition_mapping.state_dict(),
            'GVAE': self.GVAE.state_dict(),
            'decoder': self.decoder.state_dict(),
            'generator': self.generator.state_dict(),
            'optimizer': pickle.dumps(self.scheduler)
        }

        model_save_path = os.path.join(self.model_dir,
                                       'model_%d_%d_%.2f' % (self.i_epoch, self.i_step, self.current_loss))
        torch.save(state, model_save_path)
        # torch.save(state, self.best_path)

    def train_one_batch(self, batch, train=True, **unused):
        batch = {k: t.to(config.device) if type(t) != list else t for k, t in batch.items()}
        enc_batch, dec_batch = batch["input_batch"], batch["target_batch"]
        nodes = [n.to(self.device) for n in batch["nodes"]]
        node_num = batch["node_num"]
        adj_post = [adj.to(self.device) for adj in batch["adj_post"]]
        adj_prior = [adj.to(self.device) for adj in batch["adj_prior"]]
        adj_label = [adj.to(self.device) for adj in batch["adj_label"]]

        # encode input sequence
        src_mask = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        enc_hidden = self.embedding(enc_batch) + self.embedding(batch["mask_input"])
        enc_hidden = self.encoder(enc_hidden, src_mask)

        # encode the target sequence along with the latent variable
        # compute the latent variable
        next_batch = torch.cat((enc_batch, dec_batch), dim=-1)
        dec_mask = dec_batch.data.eq(config.PAD_idx) | dec_batch.data.eq(config.EOS_idx)
        next_mask_batch = torch.cat(
            (batch["mask_input"], ((1 - dec_mask.long()) * config.SYS_idx).masked_fill(dec_mask, config.PAD_idx)),
            dim=-1)
        next_mask = next_batch.data.eq(config.PAD_idx).unsqueeze(1)
        next_hidden = self.embedding(next_batch) + self.embedding(next_mask_batch)
        next_hidden = self.encoder(next_hidden, next_mask)
        random_variable = torch.randn_like(enc_hidden[:, 0:1, :])
        ctx_hidden, _ = self.hidden_mapping(random_variable, enc_hidden, enc_hidden, src_mask)
        post_hidden, _ = self.hidden_mapping(random_variable, next_hidden, next_hidden, next_mask)
        latent_seq, kld_seq = self.SeqVAE(ctx_hidden, post_hidden, train)

        # encode causal graph based on the condition including context hidden, sequence latent, emotion
        ns_embed = self.node_embedding(torch.stack(nodes, 0))
        emotion_emb = self.emotion_embedding(batch["emotion_label"])
        conds = torch.cat((ctx_hidden, emotion_emb.unsqueeze(1), latent_seq.unsqueeze(1)), dim=1)
        conds = F.leaky_relu(self.condition_mapping(conds))
        relations, relation_masks, recons_loss, kld_loss_g = [], [], [], []
        for i, (adj_r, adj_p, adj_l) in enumerate(zip(adj_post, adj_prior, adj_label)):
            rel, mask, (recons, kld_g) = self.GVAE(ns_embed[i][:node_num[i]], adj_r, adj_p, conds[i:i + 1, :, :], adj_l,
                                                   train)
            relations.append(rel)
            relation_masks.append(mask)
            recons_loss.append(recons)
            kld_loss_g.append(kld_g)

        rel_hidden = torch.stack(relations, dim=0)
        rel_mask = torch.stack(relation_masks, dim=0).unsqueeze(1)
        graph_loss = torch.sum(torch.stack(recons_loss, dim=-1), dim=0)
        kld_loss_VGAE = torch.sum(torch.stack(kld_loss_g, dim=-1), dim=0)

        # decoding
        sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1).to(self.device)
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), dim=1)
        trg_mask = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        pre_logit, _ = self.decoder(self.embedding(dec_batch_shift), (enc_hidden, rel_hidden),
                                    ((src_mask, rel_mask), trg_mask))
        logit = self.generator(pre_logit).clamp(min=1e-9, max=1 - 1e-9)

        # compute the final loss
        # other_loss = kld_loss_VGAE + torch.sum(kld_seq, dim=0)
        other_loss = kld_loss_VGAE + kld_seq
        loss_ppl = self.criterion_ppl(logit.contiguous().log().view(-1, logit.size(-1)),
                                      dec_batch.contiguous().view(-1))

        if train:
            self.i_step += 1
            rate = min(config.bz * self.attempt * (self.i_step // self.attempt) / 4e4, 1)
            token_loss = self.criterion(logit.contiguous().log().view(-1, logit.size(-1)),
                                        dec_batch.contiguous().view(-1))
            if token_loss.isnan() or token_loss.isinf():
                print('Token Loss is Nan % d Inf %d' % (token_loss.isnan(), token_loss.isinf()))
            assert not token_loss.isnan()
            assert not token_loss.isinf()
            loss = token_loss + graph_loss + other_loss * rate
            loss.backward()
            if self.i_step & self.attempt == 0:
                if config.noam:
                    self.scheduler.step()
                    self.scheduler.optimizer.zero_grad()
                else:
                    self.scheduler.step(loss_ppl)
                    self.optimizer.zero_grad()

        # return the generation loss, ppl and other loss
        return loss_ppl.item(), math.exp(min(loss_ppl, 100)), other_loss.item()

    def decoder_greedy(self, batch, max_dec_step=30):
        batch = {k: t.to(config.device) if type(t) != list else t for k, t in batch.items()}
        enc_batch = batch["input_batch"]
        nodes = [n.to(self.device) for n in batch["nodes"]]
        node_num = batch["node_num"]
        adj_prior = [adj.to(self.device) for adj in batch["adj_prior"]]

        # encode input sequence
        src_mask = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        enc_hidden = self.embedding(enc_batch) + self.embedding(batch["mask_input"])
        enc_hidden = self.encoder(enc_hidden, src_mask)

        # compute the latent variable
        random_variable = torch.randn_like(enc_hidden[:, 0:1, :])
        ctx_hidden, _ = self.hidden_mapping(random_variable, enc_hidden, enc_hidden, src_mask)
        latent_seq = self.SeqVAE.generate(ctx_hidden)

        # encode causal graph based on the condition including context hidden, sequence latent, emotion
        ns_embed = self.node_embedding(torch.stack(nodes, 0))
        emotion_emb = self.emotion_embedding(batch["emotion_label"])
        conds = torch.cat((ctx_hidden, emotion_emb.unsqueeze(1), latent_seq.unsqueeze(1)), dim=1)  #
        conds = F.leaky_relu(self.condition_mapping(conds))
        relations, relation_masks = [], []
        for i, adj in enumerate(adj_prior):
            rel, mask = self.GVAE.generate(ns_embed[i][:node_num[i]], adj, conds[i:i + 1, :])
            relations.append(rel)
            relation_masks.append(mask)
        rel_hidden = torch.stack(relations, dim=0)
        rel_mask = torch.stack(relation_masks, dim=0).unsqueeze(1)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(self.device)
        trg_mask = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            pre_logit, _ = self.decoder(self.embedding(ys), (enc_hidden, rel_hidden), ((src_mask, rel_mask), trg_mask))
            logit = self.generator(pre_logit).clamp(min=1e-9, max=1 - 1e-9)

            _, next_word = torch.max(logit[:, -1], dim=1)
            decoded_words.append(
                ['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in
                 next_word.view(-1)])
            next_word = next_word.data[0]

            ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).to(self.device)], dim=1).to(self.device)
            trg_mask = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>':
                    break
                else:
                    st += e + ' '
            sent.append(st)
        return sent
