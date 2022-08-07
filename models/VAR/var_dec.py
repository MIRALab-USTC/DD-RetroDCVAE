from numpy.core.fromnumeric import var
import torch
import torch.nn as nn

from onmt.decoders.decoder import DecoderBase
from onmt.modules import MultiHeadedAttention, AverageAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask


from models.VAR.utils import gaussian_kld, Con_Layer_Norm, VarFeedForward, FeedForwardNet, gumbel_softmax, tile
import torch.nn.functional as F

class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder layer block in Pre-Norm style.
    Pre-Norm style is an improvement w.r.t. Original paper's Post-Norm style,
    providing better converge speed and performance. This is also the actual
    implementation in tensor2tensor and also avalable in fairseq.
    See https://tunz.kr/post/4 and :cite:`DeeperTransformer`.

    .. mermaid::

        graph LR
        %% "*SubLayer" can be self-attn, src-attn or feed forward block
            A(input) --> B[Norm]
            B --> C["*SubLayer"]
            C --> D[Drop]
            D --> E((+))
            A --> E
            E --> F(out)


    Args:
        d_model (int): the dimension of keys/values/queries in
            :class:`MultiHeadedAttention`, also the input size of
            the first-layer of the :class:`PositionwiseFeedForward`.
        heads (int): the number of heads for MultiHeadedAttention.
        d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
        dropout (float): dropout in residual, self-attn(dot) and feed-forward
        attention_dropout (float): dropout in context_attn (and self-attn(avg))
        self_attn_type (string): type of self-attention scaled-dot, average
        max_relative_positions (int):
            Max distance between inputs in relative positions representations
        aan_useffn (bool): Turn on the FFN layer in the AAN decoder
        full_context_alignment (bool):
            whether enable an extra full context decoder forward for alignment
        alignment_heads (int):
            N. of cross attention heads to use for alignment guiding
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 self_attn_type="scaled-dot", max_relative_positions=0,
                 aan_useffn=False, full_context_alignment=False,
                 alignment_heads=0):
        super(TransformerDecoderLayer, self).__init__()

        if self_attn_type == "scaled-dot":
            self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=attention_dropout,
                max_relative_positions=max_relative_positions)
        elif self_attn_type == "average":
            self.self_attn = AverageAttention(d_model,
                                              dropout=attention_dropout,
                                              aan_useffn=aan_useffn)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.full_context_alignment = full_context_alignment
        self.alignment_heads = alignment_heads

    def forward(self, *args, **kwargs):
        """ Extend `_forward` for (possibly) multiple decoder pass:
        Always a default (future masked) decoder forward pass,
        Possibly a second future aware decoder pass for joint learn
        full context alignement, :cite:`garg2019jointly`.

        Args:
            * All arguments of _forward.
            with_align (bool): whether return alignment attention.

        Returns:
            (FloatTensor, FloatTensor, FloatTensor or None):

            * output ``(batch_size, T, model_dim)``
            * top_attn ``(batch_size, T, src_len)``
            * attn_align ``(batch_size, T, src_len)`` or None
        """
        with_align = kwargs.pop('with_align', False)
        output, attns = self._forward(*args, **kwargs)
        top_attn = attns[:, 0, :, :].contiguous()
        attn_align = None
        if with_align:
            if self.full_context_alignment:
                # return _, (B, Q_len, K_len)
                _, attns = self._forward(*args, **kwargs, future=True)

            if self.alignment_heads > 0:
                attns = attns[:, :self.alignment_heads, :, :].contiguous()
            # layer average attention across heads, get ``(B, Q, K)``
            # Case 1: no full_context, no align heads -> layer avg baseline
            # Case 2: no full_context, 1 align heads -> guided align
            # Case 3: full_context, 1 align heads -> full cte guided align
            attn_align = attns.mean(dim=1)
        return output, top_attn, attn_align

    def _forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                 layer_cache=None, step=None, condition=None, future=False):
        """ A naive forward pass for transformer decoder.

        # T: could be 1 in the case of stepwise decoding or tgt_len

        Args:
            inputs (FloatTensor): ``(batch_size, T, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (bool): ``(batch_size, 1, src_len)``
            tgt_pad_mask (bool): ``(batch_size, 1, T)``
            layer_cache (dict or None): cached layer info when stepwise decode
            step (int or None): stepwise decoding counter
            future (bool): If set True, do not apply future_mask.

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, T, model_dim)``
            * attns ``(batch_size, head, T, src_len)``

        """
        dec_mask = None

        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            if not future:  # apply future_mask, result mask in (B, T, T)
                future_mask = torch.ones(
                    [tgt_len, tgt_len],
                    device=tgt_pad_mask.device,
                    dtype=torch.uint8)
                future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
                # BoolTensor was introduced in pytorch 1.2
                try:
                    future_mask = future_mask.bool()
                except AttributeError:
                    pass
                dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
            else:  # only mask padding, result mask in (B, 1, T)
                dec_mask = tgt_pad_mask

        input_norm = self.layer_norm_1(inputs)

        if isinstance(self.self_attn, MultiHeadedAttention):
            query, _ = self.self_attn(input_norm, input_norm, input_norm,
                                      mask=dec_mask,
                                      layer_cache=layer_cache,
                                      attn_type="self")
        elif isinstance(self.self_attn, AverageAttention):
            query, _ = self.self_attn(input_norm, mask=dec_mask,
                                      layer_cache=layer_cache, step=step)

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, attns = self.context_attn(memory_bank, memory_bank, query_norm,
                                       mask=src_pad_mask,
                                       layer_cache=layer_cache,
                                       attn_type="context")
        output = self.feed_forward(self.drop(mid) + query)

        return output, attns

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.context_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout


class VariationalDecoderLayer(nn.Module):
    def __init__(self, args, d_model, heads, d_ff, dropout, attention_dropout,
                 self_attn_type="scaled-dot", max_relative_positions=0,
                 aan_useffn=False, full_context_alignment=False,
                 alignment_heads=0):
        super(VariationalDecoderLayer, self).__init__()

        if self_attn_type == "scaled-dot":
            self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=attention_dropout,
                max_relative_positions=max_relative_positions)
        elif self_attn_type == "average":
            self.self_attn = AverageAttention(d_model,
                                              dropout=attention_dropout,
                                              aan_useffn=aan_useffn)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout)
        # self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        hid_size = args.varFFN_hidden_size # hid_size = 128
        latent_size = args.latent_size # latent_size = 60
        vocab_size = args.vocab_size  # vocab_size = 75
        condition_size = d_model

        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)
        self.prior = VarFeedForward(d_model, hid_size, 2 * latent_size,
                                    layer_config='lll', dropout=dropout)
        self.recog = VarFeedForward(d_model * 2, hid_size, 2 * latent_size,
                                    layer_config='lll', dropout=dropout)
        self.criterion = nn.NLLLoss(ignore_index=1)
        # self.z_score2 = nn.Sequential(nn.Linear(latent_size, vocab_size), nn.LogSoftmax(dim=-1))
        # self.feed_forward = FeedForwardNet(d_model+latent_size, d_ff, d_model, dropout) # var version  d_in != d_out
        self.feed_forward = FeedForwardNet(d_model+latent_size, d_ff, d_model, dropout=0) # var version for z  d_in != d_out
        
        self.conln_1 = Con_Layer_Norm(d_model, condition_size)  #(d_model, d_model)
        self.conln_2 = Con_Layer_Norm(d_model, condition_size)
        self.drop = nn.Dropout(dropout)
        self.full_context_alignment = full_context_alignment
        self.alignment_heads = alignment_heads

    def forward(self, *args, **kwargs):
        with_align = kwargs.pop('with_align', False)
        output, attns, z, kld_loss = self._forward(*args, **kwargs)
        top_attn = attns[:, 0, :, :].contiguous()
        attn_align = None
        if with_align:
            if self.full_context_alignment:
                # return _, (B, Q_len, K_len)
                _, attns, z = self._forward(*args, **kwargs, future=True)

            if self.alignment_heads > 0:
                attns = attns[:, :self.alignment_heads, :, :].contiguous()
            # layer average attention across heads, get ``(B, Q, K)``
            # Case 1: no full_context, no align heads -> layer avg baseline
            # Case 2: no full_context, 1 align heads -> guided align
            # Case 3: full_context, 1 align heads -> full cte guided align
            attn_align = attns.mean(dim=1)
        return output, top_attn, z, kld_loss, attn_align

    def _forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                 layer_cache=None, step=None, condition=None, future=False):
        train = True if layer_cache is None else False
        dec_mask = None

        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            if not future:  # apply future_mask, result mask in (B, T, T)
                future_mask = torch.ones(
                    [tgt_len, tgt_len],
                    device=tgt_pad_mask.device,
                    dtype=torch.uint8)
                future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
                # BoolTensor was introduced in pytorch 1.2
                try:
                    future_mask = future_mask.bool()
                except AttributeError:
                    pass
                dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
            else:  # only mask padding, result mask in (B, 1, T)
                dec_mask = tgt_pad_mask

        input_norm = self.layer_norm_1(inputs)

        if isinstance(self.self_attn, MultiHeadedAttention):
            query, _ = self.self_attn(input_norm, input_norm, input_norm,
                                      mask=dec_mask,
                                      layer_cache=layer_cache,
                                      attn_type="self")
        elif isinstance(self.self_attn, AverageAttention):
            query, _ = self.self_attn(input_norm, mask=dec_mask,
                                      layer_cache=layer_cache, step=step)

        query = self.drop(query) + inputs

        query_norm = self.conln_1(query, memory_bank)
        mid, attns = self.context_attn(memory_bank, memory_bank, query_norm,
                                       mask=src_pad_mask,
                                       layer_cache=layer_cache,
                                       attn_type="context")
        mid_norm = self.conln_2(mid, memory_bank)  # type_emb  g_rep self.conln(mid, g_rep)
        prior = {"mean": [], "std": []}
        post = {"mean": [], "std": []}
        # Prior net for testing
        mu, log_var = self.prior(mid_norm).chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var)
        prior["mean"] = mu
        prior["std"] = std
        # Posterior net for training
        kld_loss = 0
        if train:
            mu, log_var = self.recog(torch.cat([mid_norm, inputs], dim=-1)).chunk(2, dim=-1)

            std = torch.exp(0.5 * log_var)
            post["mean"] = mu
            post["std"] = std
            kld_loss = gaussian_kld(post["mean"], post["std"], prior["mean"], prior["std"])  # print(kld_loss[:3, :4])
            kld_loss = torch.mean(kld_loss)  # original

        # reparameterize
        eps = torch.randn(size=mu.size(), device=mu.device)
        z = eps * std + mu

        # Positionwise Feedforward
        output = self.feed_forward(torch.cat([z, mid_norm], dim=-1))  # torch.Size([640, 1, 256])
        output = self.drop(output) + mid  # torch.Size([141, 47, 256])

        return output, attns, z, kld_loss

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.context_attn.update_dropout(attention_dropout)
        # self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout


class VariationalDecoder(DecoderBase):
    def __init__(self, args, num_layers, d_model, heads, d_ff,
                 copy_attn, self_attn_type, dropout, attention_dropout,
                 embeddings, max_relative_positions, aan_useffn,
                 full_context_alignment, alignment_layer,
                 alignment_heads):
        super(VariationalDecoder, self).__init__()

        self.embeddings = embeddings

        # Decoder State
        self.state = {}

        var_layers = args.variational_num_layers
        num_layers = num_layers - var_layers
        self.transformer_layers = nn.ModuleList()
        for i in range(var_layers):
            self.transformer_layers.append(VariationalDecoderLayer(args, d_model, heads, d_ff, dropout,
             attention_dropout, self_attn_type=self_attn_type,
             max_relative_positions=max_relative_positions,
             aan_useffn=aan_useffn,
             full_context_alignment=full_context_alignment,
             alignment_heads=alignment_heads))
        for i in range(num_layers):
            self.transformer_layers.append(TransformerDecoderLayer(d_model, heads, d_ff, dropout,
             attention_dropout, self_attn_type=self_attn_type,
             max_relative_positions=max_relative_positions,
             aan_useffn=aan_useffn,
             full_context_alignment=full_context_alignment,
             alignment_heads=alignment_heads))

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.alignment_layer = alignment_layer

        # ++++++++++++++++ init GRU
        K = args.latent_K
        self.latent_emb = nn.Embedding(K, args.embed_size)
        self.prior = nn.GRU(d_model, hidden_size=K, num_layers=1, batch_first=True)
        self.recog = nn.GRU(d_model, hidden_size=K, num_layers=1, batch_first=True)
        self.gru = True
        self.cvae = True
        self.args = args
        

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions,
            opt.aan_useffn,
            opt.full_context_alignment,
            opt.alignment_layer,
            alignment_heads=opt.alignment_heads)

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, product_emb, tgt, memory_bank, step=None, **kwargs):
        """Decode, possibly stepwise."""
        if step == 0:
            self._init_cache(memory_bank)

        tgt_words = tgt[:, :, 0].transpose(0, 1)  # [max_tgt_t, b, 1] => [b, max_tgt_t]

        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim  [max_tgt_t, b, h]

        output = emb.transpose(0, 1).contiguous()  # [max_tgt_t, b, h] => [b, max_tgt_t, h]
        src_memory_bank = memory_bank.transpose(0, 1).contiguous() # [max_src_t, b, h] => [b, max_src_t, h]
        product_emb = product_emb.transpose(0, 1).contiguous() # [max_src_t, b, h] => [b, max_src_t, h]
        #++++++++++++++++++++ Gumbel-Softmax
        kld_loss = 0
        ## for CVAE
        if hasattr(self, 'cvae') and (step is None or step==0):
            if hasattr(self, 'gru'):
                logits = self.prior(src_memory_bank, None)[-1][0] #  GRU
            else:
                logits = self.prior(src_memory_bank.mean(dim=1))
            prior = F.log_softmax(logits, dim=-1)
            if self.training:
                logits = self.recog(torch.cat([src_memory_bank, output], dim=1), None)[-1][0] #  GRU
                recog = F.softmax(logits, dim=-1)
                kld_loss = F.kl_div(prior, recog, reduction="sum")
            y_sample = gumbel_softmax(logits, hard=True)
            ## for beam search during inference, i.e., perform validate.sh and predict.sh
            beam_size = self.args.beam_size
            if not self.training and beam_size>5:
                y_sample = torch.stack(y_sample.split(beam_size, dim=0), dim=1)[0]  # [b, h]
                y_sample = tile(y_sample, count=beam_size, dim=0)
            latent = y_sample @ self.latent_emb.weight
            output[:,0] += latent
            self.y_sample = y_sample

        pad_idx = self.embeddings.word_padding_idx
        src_lens = kwargs["memory_lengths"]
        src_max_len = self.state["src"].shape[0]
        src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        with_align = kwargs.pop('with_align', False)
        attn_aligns = []

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            others = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step,
                with_align=with_align)
            if layer._get_name() == 'TransformerDecoderLayer':
                output, attn, attn_align = others
            elif layer._get_name() == 'VariationalDecoderLayer':
                output, attn, z, kld_loss, attn_align = others
            if attn_align is not None:
                attn_aligns.append(attn_align)
        
        aux_loss = 0
        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn
        if with_align:
            attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
            # attns["align"] = torch.stack(attn_aligns, 0).mean(0)  # All avg

        # TODO change the way attns is returned dict => list or tuple (onnx)
        if step is None: ## train or validate
            return dec_outs, attns, aux_loss, kld_loss
        else:
            return dec_outs, attns

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            if isinstance(layer.self_attn, AverageAttention):
                layer_cache["prev_g"] = torch.zeros((batch_size, 1, depth),
                                                    device=memory_bank.device)
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)
