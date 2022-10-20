### MOST OF IT TAKEN FROM https://github.com/kolloldas/torchnlp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import pickle
from utils import config
from utils.metric import moses_multi_bleu

from utils.beam_omt_graph import Translator
import pprint
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=1)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


class EncoderLayer(nn.Module):
    """
    Represents one Encoder layer of the Transformer Encoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(
            self,
            hidden_size,
            total_key_depth,
            total_value_depth,
            filter_size,
            num_heads,
            bias_mask=None,
            layer_dropout=0.2,
            attention_dropout=0.2,
            relu_dropout=0.2
    ):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            bias_mask,
            attention_dropout
        )

        self.positionwise_feed_forward = PositionwiseFeedForward(
            hidden_size,
            filter_size,
            hidden_size,
            layer_config='cc',
            padding='both',
            dropout=relu_dropout
        )
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs, mask=None):
        x = inputs

        # Layer Normalization
        x_norm = self.layer_norm_mha(x)

        # Multi-head attention
        y, _ = self.multi_head_attention(x_norm, x_norm, x_norm, mask)
        # y: (batch_size, seq_length, hidden_size)
        # Dropout and residual
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual
        y = self.dropout(x + y)

        return y


class MemEncoderLayer(nn.Module):
    def __init__(
            self,
            hidden_size,
            total_key_depth,
            total_value_depth,
            filter_size,
            num_heads,
            bias_mask=None,
            layer_dropout=0.2,
            attention_dropout=0.2,
            relu_dropout=0.2
    ):
        super(MemEncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            bias_mask,
            attention_dropout
        )

        self.latent_memory_unit = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            bias_mask,
            attention_dropout
        )

        self.positionwise_feed_forward = PositionwiseFeedForward(
            hidden_size,
            filter_size,
            hidden_size,
            layer_config='cc',
            padding='both',
            dropout=relu_dropout
        )
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_lmu = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        x, memory, attention_weight, mask = inputs
        mask_memory, mask_src = mask

        # Layer Normalization
        x_norm = self.layer_norm_mha(x)

        # Multi-head attention
        y, _ = self.multi_head_attention(x_norm, x_norm, x_norm, mask_src)

        # Dropout and residual
        x = self.dropout(x + y)

        # Layer Normalization before latent memory unit
        x_norm = self.layer_norm_lmu(x)

        # Latent Memory Unit
        key_value = torch.cat((memory, x_norm), dim=1)
        kv_mask = torch.cat((mask_memory, mask_src), dim=-1)
        y, attention_weight = self.latent_memory_unit(x_norm, key_value, key_value, kv_mask)

        # Dropout and residual after latent memory unit
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual after positionwise feed forward layer
        y = self.dropout(x + y)

        # Return encoder outputs as well to work with nn.Sequential
        return y, memory, attention_weight, mask


class DecoderLayer(nn.Module):
    """
    Represents one Decoder layer of the Transformer Decoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(
            self,
            hidden_size,
            total_key_depth,
            total_value_depth,
            filter_size,
            num_heads,
            bias_mask,
            layer_dropout=0.2,
            attention_dropout=0.2,
            relu_dropout=0.2
    ):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(DecoderLayer, self).__init__()

        self.multi_head_attention_dec = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            bias_mask,
            attention_dropout
        )

        self.multi_head_attention_enc_dec = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            None,
            attention_dropout
        )

        self.positionwise_feed_forward = PositionwiseFeedForward(
            hidden_size,
            filter_size,
            hidden_size,
            layer_config='cc',
            padding='both' if bias_mask is None else 'left',
            dropout=relu_dropout
        )
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(hidden_size)
        self.layer_norm_mha_enc = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        """
        NOTE: Inputs is a tuple consisting of decoder inputs and encoder output
        """
        x, encoder_outputs, attention_weight, mask = inputs
        mask_src, dec_mask = mask

        # Layer Normalization before decoder self attention
        x_norm = self.layer_norm_mha_dec(x)

        # Masked Multi-head attention
        y, _ = self.multi_head_attention_dec(x_norm, x_norm, x_norm, dec_mask)

        # Dropout and residual after self-attention
        x = self.dropout(x + y)

        # Layer Normalization before encoder-decoder attention
        x_norm = self.layer_norm_mha_enc(x)

        # Multi-head encoder-decoder attention
        y, attention_weight = self.multi_head_attention_enc_dec(x_norm, encoder_outputs, encoder_outputs, mask_src)

        # Dropout and residual after encoder-decoder attention
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual after positionwise feed forward layer
        y = self.dropout(x + y)

        # y = self.layer_norm_end(y)

        # Return encoder outputs as well to work with nn.Sequential
        return y, encoder_outputs, attention_weight, mask


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """

    def __init__(
            self,
            input_depth,
            total_key_depth,
            total_value_depth,
            output_depth,
            num_heads,
            bias_mask=None,
            dropout=0.0
    ):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

        if total_key_depth % num_heads != 0:
            print("Key depth (%d) must be divisible by the number of "
                  "attention heads (%d)." % (total_key_depth, num_heads))
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            print("Value depth (%d) must be divisible by the number of "
                  "attention heads (%d)." % (total_value_depth, num_heads))
            total_value_depth = total_value_depth - (total_value_depth % num_heads)

        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5  ## sqrt
        self.bias_mask = bias_mask

        # Key and query depth will be same
        if type(input_depth) == tuple:
            input_depth_q, input_depth_k, input_depth_v = input_depth
        else:
            input_depth_q, input_depth_k, input_depth_v = input_depth, input_depth, input_depth

        self.query_linear = nn.Linear(input_depth_q, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth_k, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth_v, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3] * self.num_heads)

    def forward(self, queries, keys, values, mask):
        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)

        # Split into multiple heads
        queries = self._split_heads(queries)  # (batch_size, num_heads, seq_length, d)
        keys = self._split_heads(keys)  # (batch_size, num_heads, seq_length_k, d)
        values = self._split_heads(values)  # (batch_size, num_heads, seq_length_v, d)

        # Scale queries
        queries = queries * self.query_scale

        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        # shape: (batch_size, num_heads, seq_length_q, seq_length_k)

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            logits = logits.masked_fill_(mask, -1e18)

        ## attention weights
        attetion_weights = logits.sum(dim=1) / self.num_heads  # (batch_size, seq_length, seq_length)

        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)  # (batch_size, num_heads, seq_length, seq_length)
        # Dropout
        weights = self.dropout(weights)

        # Combine with values to get context
        contexts = torch.matmul(weights, values)
        # shapes: (batch_size, num_heads, seq_length, seq_length), (batch_size, num_heads, seq_length, depth / num_heads)
        # --> (batch_size, num_heads, seq_length, depth / num_heads)

        # Merge heads
        contexts = self._merge_heads(contexts)
        # shape: (batch_size, seq_length, depth)

        # Linear to get output
        outputs = self.output_linear(contexts)
        # shape: (batch_size, seq_length, output_size)
        return outputs, attetion_weights


class Conv(nn.Module):
    """
    Convenience class that does padding and convolution for inputs in the format
    [batch_size, sequence length, hidden size]
    """

    def __init__(
            self,
            input_size,
            output_size,
            kernel_size,
            pad_type
    ):
        """
        Parameters:
            input_size: Input feature size
            output_size: Output feature size
            kernel_size: Kernel width
            pad_type: left -> pad on the left side (to mask future data),
                      both -> pad on both sides
        """
        super(Conv, self).__init__()
        padding = (kernel_size - 1, 0) if pad_type == 'left' else (kernel_size // 2, (kernel_size - 1) // 2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        inputs = self.pad(inputs.permute(0, 2, 1))
        outputs = self.conv(inputs).permute(0, 2, 1)

        return outputs


class PositionwiseFeedForward(nn.Module):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """

    def __init__(
            self,
            input_depth,
            filter_size,
            output_depth,
            layer_config='ll',
            padding='left',
            dropout=0.0
    ):
        """
        Parameters:
            input_depth: Size of last dimension of input
            filter_size: Hidden size of the middle layer
            output_depth: Size last dimension of the final output
            layer_config: ll -> linear + ReLU + linear
                          cc -> conv + ReLU + conv etc.
            padding: left -> pad on the left side (to mask future data),
                     both -> pad on both sides
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(PositionwiseFeedForward, self).__init__()

        layers = []
        sizes = ([(input_depth, filter_size)] +
                 [(filter_size, filter_size)] * (len(layer_config) - 2) +
                 [(filter_size, output_depth)])

        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            elif lc == 'c':
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)

        return x


class LayerNorm(nn.Module):
    # Borrowed from jekbradbury
    # https://github.com/pytorch/pytorch/issues/1959
    def __init__(
            self,
            features,
            eps=1e-6
    ):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)
    # (max_length, max_length)
    return torch_mask.unsqueeze(0).unsqueeze(1)
    # (1, 1, max_length, max_length)


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1)
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    # shapes: (length, 1); (1, channels // 2) --> (length, channels // 2)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    # shape: (length, 2 * channels // 2)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 'constant', constant_values=[0.0, 0.0])
    # shape: (length, channels)
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


def _get_attn_subsequent_mask(size):
    """
    Get an attention mask to avoid using the subsequent info.
    Args:
        size: int
    Returns:
        (`LongTensor`):
        * subsequent_mask `[1 x size x size]`
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)

    return subsequent_mask.to(config.device)


class OutputLayer(nn.Module):
    """
    Abstract base class for output layer.
    Handles projection to output labels
    """

    def __init__(self, hidden_size, output_size):
        super(OutputLayer, self).__init__()
        self.output_size = output_size
        self.output_projection = nn.Linear(hidden_size, output_size)

    def loss(self, hidden, labels):
        raise NotImplementedError('Must implement {}.loss'.format(self.__class__.__name__))


class SoftmaxOutputLayer(OutputLayer):
    """
    Implements a softmax based output layer
    """

    def forward(self, hidden):
        logits = self.output_projection(hidden)
        probs = F.softmax(logits, -1)
        _, predictions = torch.max(probs, dim=-1)

        return predictions

    def loss(self, hidden, labels):
        logits = self.output_projection(hidden)
        log_probs = F.log_softmax(logits, -1)
        return F.nll_loss(log_probs.view(-1, self.output_size), labels.view(-1))


def position_encoding(sentence_size, embedding_dim):
    encoding = np.ones((embedding_dim, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_dim + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (embedding_dim + 1) / 2) * (j - (sentence_size + 1) / 2)
    encoding = 1 + 4 * encoding / embedding_dim / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    # encoding[:, -1] = 1.0
    return np.transpose(encoding)


def gen_embeddings(vocab, embed_path=None):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = np.random.randn(vocab.n_words, config.embed_dim) * 0.01
    print('Embeddings: %d x %d' % (vocab.n_words, config.embed_dim))
    if embed_path is None:
        embed_path = config.embed_path
    if os.path.exists(embed_path):
        embeddings = pickle.load(open(embed_path, "rb"))
    elif config.glove_embed is not None:
        print('Loading embedding file: %s' % config.glove_embed)
        pre_trained = 0
        for line in open(config.glove_embed, 'r', encoding='UTF8').readlines():
            sp = line.split()
            if (len(sp) == config.embed_dim + 1):
                if sp[0] in vocab.word2index:
                    pre_trained += 1
                    embeddings[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print(sp[0])
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / vocab.n_words))
        pickle.dump(embeddings, open(embed_path, 'wb'))
    return embeddings


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model, padding_idx=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


def share_embedding(vocab, pretrain=True, embed_path=None):
    embedding = Embeddings(vocab.n_words, config.embed_dim, padding_idx=config.PAD_idx)
    if pretrain:
        pre_embedding = gen_embeddings(vocab, embed_path)
        embedding.lut.weight.data.copy_(torch.FloatTensor(pre_embedding))
        embedding.lut.weight.data.requires_grad = True
    return embedding


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.size()[0] > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def state_dict(self):
        return self.optimizer.state_dict()

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(config.PAD_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


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


def print_custum(emotion, dial, ref, hyp_b):
    print("\n\nemotion:{}".format(emotion))
    print()
    print("Context:\n***********************************************************************\n*"
          "{}\n***********************************************************************".format('\n*'.join(dial)))
    print()
    print("Beam: {}".format(hyp_b))
    print()
    print("Ref: {}".format(ref))
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")


def plot_ptr_stats(model):
    stat_dict = model.generator.stats
    a = np.mean(stat_dict["a"])
    a_1_g = np.mean(stat_dict["a_1_g"])
    a_1_g_1 = np.mean(stat_dict["a_1_g_1"])
    a_STD = np.std(stat_dict["a"])
    a_1_g_STD = np.std(stat_dict["a_1_g"])
    a_1_g_1_STD = np.std(stat_dict["a_1_g_1"])
    name = ['Vocab', 'Dialg', 'DB']
    x_pos = np.arange(3)
    CTEs = [a, a_1_g, a_1_g_1]
    error = [a_STD, a_1_g_STD, a_1_g_1_STD]
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Distribution weights')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(name)
    ax.yaxis.grid(True)
    # Save the figure and show
    plt.tight_layout()
    plt.savefig(config.save_path + 'bar_plot_with_error_bars.png')


def evaluate(model, data_loader, ty='valid', max_dec_step=30, save=False):
    emotion_lst, batch_lst, ref, hyp_b = [], [], [], []
    if ty == "test":
        print("testing generation:")
    t = Translator(model, model.vocab)
    l = []
    p = []
    kld = []
    pbar = tqdm(data_loader, desc="Evaluating", leave=True, position=0)
    for j, batch in enumerate(pbar):
        loss, ppl, kld_loss = model.train_one_batch(batch, train=False)
        l.append(loss)
        p.append(ppl)
        kld.append(kld_loss)
        if ty == "test":
            sent_b = t.beam_search(batch, max_dec_step=max_dec_step)
            for i, beam_sent in enumerate(sent_b):
                emotion_lst.append(batch["emotion_txt"][i])
                batch_lst.append([" ".join(s) for s in batch['input_txt'][i]])
                rf = " ".join(batch['target_txt'][i])
                hyp_b.append(beam_sent)
                ref.append(rf)
                print_custum(
                    emotion=batch["emotion_txt"][i],
                    dial=[" ".join(s) for s in batch['input_txt'][i]],
                    ref=rf,
                    hyp_b=beam_sent
                )
            pbar.set_description("\nloss:{:.4f} ppl:{:.1f}".format(np.mean(l), math.exp(np.mean(l))))
    pbar.close()
    if ty == "test" and save:
        hyp_b_pd = pd.DataFrame(hyp_b)
        if config.max_k == -1:
            hyp_b_pd.to_csv(config.save_path + '/test/{}_beam.txt'.format(model.i_step), index=False, header=False)
        else:
            hyp_b_pd.to_csv(config.save_path + '/test/{}_{}_beam.txt'.format(model.i_step, config.max_k), index=False, header=False)
    loss = np.mean(l)
    kld = np.mean(kld)
    bleu_score_b = moses_multi_bleu(np.array(hyp_b), np.array(ref), lowercase=True)

    print("{}{}{}{}{}".format('EVAL'.ljust(10), 'Loss'.ljust(10), 'PPL'.ljust(10), 'OtherLoss'.ljust(10),
                              'Bleu_b'.ljust(10)))
    print("{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}".format(ty, loss, math.exp(loss), kld, bleu_score_b))

    return loss, math.exp(loss), kld, bleu_score_b


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x


def top_k_top_p_filtering(logits, top_k=0, top_p=0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits
