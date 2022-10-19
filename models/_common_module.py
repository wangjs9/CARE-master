from utils import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from models._common_layer import EncoderLayer, MemEncoderLayer, DecoderLayer, LayerNorm, _gen_bias_mask, \
    _gen_timing_signal, _get_attn_subsequent_mask


class Encoder(nn.Module):
    def __init__(
            self,
            embedding_size,
            hidden_size,
            num_layers,
            num_heads,
            total_key_depth,
            total_value_depth,
            filter_size,
            max_length=1024,
            input_dropout=0.2,
            layer_dropout=0.2,
            attention_dropout=0.2,
            relu_dropout=0.2,
            use_mask=False,
            universal=False
    ):
        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length) if use_mask else None,
            layer_dropout,
            attention_dropout,
            relu_dropout
        )

        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

        if config.act:
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)  # (batch_size, seq_len, embed_dim)
        # Project to hidden size
        x = self.embedding_proj(x)  # (batch_size, seq_len, hidden_size)

        if self.universal:
            if config.act:
                x, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.enc,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers
                )
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x = self.enc(x, mask=mask)
                # x = torch.mul(self.enc(x, mask=mask), cazprob + 1)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)

        return y


class Attn_Encoder(nn.Module):
    def __init__(
            self,
            embedding_size,
            hidden_size,
            num_layers,
            num_heads,
            total_key_depth,
            total_value_depth,
            filter_size,
            max_length=1024,
            input_dropout=0.2,
            layer_dropout=0.2,
            attention_dropout=0.2,
            relu_dropout=0.2,
            use_mask=False,
            universal=False
    ):

        super(Attn_Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        self.latent_proj = nn.Linear(hidden_size, hidden_size * num_layers, bias=False)

        if self.universal:
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length) if use_mask else None,
            layer_dropout,
            attention_dropout,
            relu_dropout
        )

        if self.universal:
            self.enc = DecoderLayer(*params)
        else:
            self.enc = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

        if config.act:
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None

    def forward(self, inputs, attn_output, mask, latent=None):
        mask_attn, mask_src = mask

        # Add input dropout
        x = self.input_dropout(inputs)  # (batch_size, seq_len, embed_dim)
        # Project to hidden size
        x = self.embedding_proj(x)  # (batch_size, seq_len, hidden_size)

        if latent is not None:
            # Split latent
            latent = self.latent_proj(latent).view(-1, self.num_layers, self.hidden_size)
            mask_one = torch.ones_like(mask_src.index_select(-1, torch.tensor([0], device=config.device)))
            mask_attn = torch.cat((mask_one, mask_attn), dim=-1)

        if self.universal:
            if config.act:
                x, (self.reminders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.enc,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                    attn_output,
                    decoding=True
                )
                y = self.layer_norm(x)

            else:
                for l in range(self.num_layers):
                    if latent is None:
                        attn_output_input = attn_output
                    else:
                        attn_output_input = torch.cat(
                            (latent.index_select(1, torch.tensor([l], device=config.device)), attn_output), dim=1)
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x, _, attn_dist, _ = self.enc((x, attn_output_input, [], (mask_attn, mask_src)))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            for l in range(self.num_layers):
                if latent is None:
                    attn_output_input = attn_output
                else:
                    attn_output_input = torch.cat(
                        (latent.index_select(1, torch.tensor([l], device=config.device)), attn_output), dim=1)
                x, _, attn_dist, _ = self.enc((x, attn_output_input, [], (mask_attn, mask_src)))

            # Final layer normalization
            y = self.layer_norm(x)
        return y


class MemEncoder(nn.Module):
    def __init__(
            self,
            embedding_size,
            hidden_size,
            num_layers,
            num_heads,
            total_key_depth,
            total_value_depth,
            filter_size,
            max_length=1024,
            input_dropout=0.2,
            layer_dropout=0.2,
            attention_dropout=0.2,
            relu_dropout=0.2,
            use_mask=False,
            universal=False
    ):
        super(MemEncoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        self.latent_proj = nn.Linear(hidden_size, hidden_size * num_layers, bias=False)

        if self.universal:
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length) if use_mask else None,
            layer_dropout,
            attention_dropout,
            relu_dropout
        )
        if self.universal:
            self.enc = MemEncoderLayer(*params)
        else:
            self.enc = nn.Sequential(*[MemEncoderLayer(*params) for l in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

        if config.act:
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None

    def forward(self, inputs, attn_output, mask, latent=None):
        mask_attn, mask_src = mask

        # Add input dropout
        x = self.input_dropout(inputs)  # (batch_size, seq_len, embed_dim)
        # Project to hidden size
        x = self.embedding_proj(x)  # (batch_size, seq_len, hidden_size)

        if latent is not None:
            # Split latent
            latent = self.latent_proj(latent).view(-1, self.num_layers, self.hidden_size)
            mask_one = torch.ones_like(mask_src.index_select(-1, torch.tensor([0], device=config.device)))
            mask_attn = torch.cat((mask_one, mask_attn), dim=-1)

        if self.universal:
            if config.act:
                x, (self.reminders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.enc,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                    attn_output,
                    decoding=True
                )
                y = self.layer_norm(x)

            else:
                for l in range(self.num_layers):
                    if latent is None:
                        attn_output_input = attn_output
                    else:
                        attn_output_input = torch.cat(
                            (latent.index_select(1, torch.tensor([l], device=config.device)), attn_output), dim=1)
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x, _, attn_dist, _ = self.enc((x, attn_output_input, [], (mask_attn, mask_src)))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            for l in range(self.num_layers):
                if latent is None:
                    attn_output_input = attn_output
                else:
                    attn_output_input = torch.cat(
                        (latent.index_select(1, torch.tensor([l], device=config.device)), attn_output), dim=1)
                x, _, attn_dist, _ = self.enc((x, attn_output_input, [], (mask_attn, mask_src)))

            # Final layer normalization
            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
            self,
            embedding_size,
            hidden_size,
            num_layers,
            num_heads,
            total_key_depth,
            total_value_depth,
            filter_size,
            max_length=1024,
            input_dropout=0.2,
            layer_dropout=0.2,
            attention_dropout=0.2,
            relu_dropout=0.2,
            universal=False
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),  # mandatory
            layer_dropout,
            attention_dropout,
            relu_dropout
        )

        if self.universal:
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

        if config.act:
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None

    def forward(self, inputs, encoder_output, mask):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.dec,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                    encoder_output,
                    decoding=True
                )
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist


class CVAE_Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
            self,
            embedding_size,
            hidden_size,
            num_layers,
            num_heads,
            total_key_depth,
            total_value_depth,
            filter_size,
            max_length=1024,
            input_dropout=0.2,
            layer_dropout=0.2,
            attention_dropout=0.2,
            relu_dropout=0.2,
            universal=False
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(CVAE_Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        self.latent_proj = nn.Linear(hidden_size, hidden_size * num_layers, bias=False)

        if self.universal:
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),  # mandatory
            layer_dropout,
            attention_dropout,
            relu_dropout
        )

        if self.universal:
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

        if config.act:
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None

    def forward(self, inputs, encoder_output, mask, latent):
        mask_src, mask_trg = mask
        mask_one = torch.ones_like(mask_src.index_select(-1, torch.tensor([0], device=config.device)))
        mask_src = torch.cat((mask_one, mask_src), dim=-1)
        dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        # Split latent
        latent = self.latent_proj(latent).view(-1, self.num_layers, self.hidden_size)

        if self.universal:
            if config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.dec,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                    encoder_output,
                    decoding=True
                )
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    enc_output_input = torch.cat(
                        (latent.index_select(1, torch.tensor([l], device=config.device)), encoder_output), dim=1)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x, _, attn_dist, _ = self.dec((x, enc_output_input, [], (mask_src, dec_mask)))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            for l in range(self.num_layers):
                enc_output_input = torch.cat(
                    (latent.index_select(1, torch.tensor([l], device=config.device)), encoder_output), dim=1)
                x, _, attn_dist, _ = self.dec[l]((x, enc_output_input, [], (mask_src, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(x)
        return y, attn_dist


class Multi_Source_DecoderLayer(nn.Module):
    def __init__(self, hidden_size, params):
        super(Multi_Source_DecoderLayer, self).__init__()
        self.dec_enc = DecoderLayer(*params)
        self.dec_rel = DecoderLayer(*params)
        self.merge = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, inputs):
        x, sources, attn_dist, masks = inputs
        encoder_output, rel_output = sources
        (mask_src, mask_rel), dec_mask = masks
        x_enc, _, attn_dist_enc, _ = self.dec_enc((x, encoder_output, [], (mask_src, dec_mask)))
        x_rel, _, attn_dist_rel, _ = self.dec_rel((x, rel_output, [], (mask_rel, dec_mask)))
        x = self.merge(torch.cat((x_enc, x_rel), dim=-1))
        return x, sources, (attn_dist_enc, attn_dist_rel), masks


class Graph_Infused_Decoder(nn.Module):
    def __init__(
            self,
            embedding_size,
            hidden_size,
            num_layers,
            num_heads,
            total_key_depth,
            total_value_depth,
            filter_size,
            max_length=1024,
            input_dropout=0.2,
            layer_dropout=0.2,
            attention_dropout=0.2,
            relu_dropout=0.2,
            universal=False
    ):
        super(Graph_Infused_Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),  # mandatory
            layer_dropout,
            attention_dropout,
            relu_dropout
        )

        if self.universal:
            self.dec = Multi_Source_DecoderLayer(hidden_size, params)
        else:
            self.dec = nn.Sequential(*[Multi_Source_DecoderLayer(hidden_size, params) for l in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

        if config.act:
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None

    def forward(self, inputs, sources, masks):
        source_mask, mask_trg = masks
        dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)

        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.dec,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                    sources,
                    decoding=True
                )
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x, _, attn_dist, _ = self.dec((x, sources, [], (source_mask, dec_mask)))
                y = self.layer_norm(x)

        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, sources, [], (source_mask, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        logit = self.proj(x)
        return F.softmax(logit, dim=-1)


class ACT_basic(nn.Module):
    def __init__(self, hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size, 1)
        self.p.bias.data.fill_(1)
        self.threshold = 1 - 0.1

    def forward(self, state, inputs, fn, time_enc, pos_enc, max_hop, encoder_output=None, latent=None, decoding=False):
        # init_hdd
        ## [B, S]
        halting_probability = torch.zeros(inputs.shape[0], inputs.shape[1]).to(config.device)
        ## [B, S]
        remainders = torch.zeros(inputs.shape[0], inputs.shape[1]).to(config.device)
        ## [B, S]
        n_updates = torch.zeros(inputs.shape[0], inputs.shape[1]).to(config.device)
        ## [B, S, HDD]
        previous_state = torch.zeros_like(inputs).to(config.device)

        step = 0
        # for l in range(self.num_layers):
        while (((halting_probability < self.threshold) & (n_updates < max_hop)).byte().any()):
            # as long as there is a True value, the loop continues
            # Add timing signal
            state = state + time_enc[:, :inputs.shape[1], :].type_as(inputs.data)
            state = state + pos_enc[:, step, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)

            p = self.sigma(self.p(state)).squeeze(-1)  # (1, 1)
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            if decoding:
                if latent == None:
                    state, _, attention_weight = fn((state, encoder_output, []))
                else:
                    enc_output_input = torch.cat(
                        (latent.index_select(1, torch.tensor([max_hop], device=config.device)), encoder_output), dim=1)
                    state, _, attention_weight = fn((state, enc_output_input, []))
            else:
                # apply transformation on the state
                state = fn(state)

            # update running part in the weighted state and keep the rest
            previous_state = ((state * (update_weights.unsqueeze(-1) + 1e-10)) + (
                    previous_state * (1 - update_weights.unsqueeze(-1) + 1e-10)))
            if decoding:
                if step == 0:
                    previous_att_weight = torch.zeros_like(attention_weight).to(config.device)  ## [B, S, src_size]
                previous_att_weight = ((attention_weight * update_weights.unsqueeze(-1)) + (
                        previous_att_weight * (1 - update_weights.unsqueeze(-1))))
            ## previous_state is actually the new_state at end of hte loop
            ## to save a line I assigned to previous_state so in the next
            ## iteration is correct. Notice that indeed we return previous_state
            step += 1

        if decoding:
            return previous_state, previous_att_weight, (remainders, n_updates)
        else:
            return previous_state, (remainders, n_updates)
