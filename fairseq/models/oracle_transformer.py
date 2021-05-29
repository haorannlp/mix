# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy
import nltk
import random
import math
import copy
from typing import Any, Dict, List, Optional, Tuple

import heapq

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from torch import Tensor
from typing import List, NamedTuple, Optional

EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Tensor),  # B x T
        ("encoder_embedding", Tensor),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
    ],
)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("oracle_transformer")
class TransformerModel(FairseqEncoderDecoderModel):
    @classmethod
    def hub_models(cls):
        return {}

    pad_idx = 1
    bos_idx = 2

    def __init__(self, args, encoder, decoder, generator=None):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = False
        self.use_word_oracle = args.use_word_level_oracles
        self.use_sentence_oracle = args.use_sentence_level_oracles
        self.generator = generator
        self.updates = 0
        self.epoch = 0
        self.decay_k = args.decay_k
        self.epoch_decay = args.use_epoch_numbers_decay
        self.use_greedy_gumbel_noise = args.use_greedy_gumbel_noise
        self.gumbel_noise = args.gumbel_noise
        self.use_bleu_gumbel_noise = args.use_bleu_gumbel_noise
        self.probs = 0
        
        self.use_sentence_oracle_mask = args.use_sentence_oracle_mask

        self.eval_embedding = []

        self.change_decay_prob_till = args.change_decay_prob_till

        self.use_neighbor_MLE = args.use_neighbor_MLE


        self.exponential = args.ss_exponential

        self.random_sampling = args.random_sampling_strategy

        self.topk_sampling = args.use_topk_sampling
        self.topk_k = args.topk_k

        self.random_or_greedy_sampling = args.random_or_greedy_sampling
        self.random_greedy_ratio = args.random_greedy_ratio


        self.random_mix_mle_test = args.random_mix_mle_test
        self.greedy_mix_mle_test = args.greedy_mix_mle_test
        self.no_mix_gold = args.no_mix_gold
        self.replace_non_gold_greedy = args.replace_non_gold_greedy

        self.random_sampling_greedy_output = args.random_sampling_greedy_output
        self.random_sampling_random_output = args.random_sampling_random_output
        self.word_oracle_noise_greedy_output = args.word_oracle_noise_greedy_output
        self.topk_greedy_output = args.topk_greedy_output
        self.word_oracle_no_noise_greedy_mix_random_output = args.word_oracle_no_noise_greedy_mix_random_output
        self.word_oracle_replace_non_gold_greedy_with_topk = args.word_oracle_replace_non_gold_greedy_with_topk

        self.decay_update = args.decay_update
        


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        parser.add_argument('--layer-wise-attention', default=False, action='store_true',
                            help='perform layer-wise attention (cross-attention or cross+self-attention)')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        # oracle arguments
        parser.add_argument('--use-sentence-level-oracles', action='store_true', default=False,
                            help='use sentences level oracles')
        parser.add_argument('--use-word-level-oracles', action='store_true', default=False,
                            help='use word level oracles')
        parser.add_argument('--decay-k', type=int, metavar='D', default=0,
                            help='decay k')

        parser.add_argument('--change-decay-prob-till', type=int, default=0)                       # !!!

        parser.add_argument('--use-epoch-numbers-decay', action='store_true', default=False,
                            help='probability decay by epoch number')
        parser.add_argument('--use-greedy-gumbel-noise', action='store_true', default=False,
                            help='select word with gumbel noise')
        parser.add_argument('--gumbel-noise', type=float, metavar='D', default=0.5,
                            help='word noise')
        parser.add_argument('--use-bleu-gumbel-noise', action='store_true', default=False,
                            help='generate sentence with gumbel noise')
        parser.add_argument('--oracle-search-beam-size', type=int, metavar='N', default=4,
                            help='generate oracle sentence beam size')
        parser.add_argument('--use-sentence-oracle-mask', action='store_true', default=False)
        
        parser.add_argument('--use_neighbor_MLE', action='store_true', default=False)
        parser.add_argument('--super_neighbor_MLE', action='store_true', default=False)

        parser.add_argument('--use_neighborhood_sampling', action='store_true', default=False)
        parser.add_argument('--neighborhood_distance', type=int, default=0)
        parser.add_argument('--ss-exponential', type=float, default=0.5)

        parser.add_argument('--fixed_neighborhood', action='store_true', default=False)

        parser.add_argument('--neighborhood_direction', type=str, default='down')

        parser.add_argument('--neighborhood_sampling_by_score', action='store_true', default=False)
        parser.add_argument('--increase_every', type=int, default=-1)
        parser.add_argument('--neighbor_temperature', type=float, default=0)
        #parser.add_argument('--neighbor_range', type=str, default='range',
        #                    help='fixed_position, range')

        parser.add_argument('--random_sampling_strategy', action='store_true', default=False)

        parser.add_argument('--use-topk-sampling', action='store_true', default=False)
        parser.add_argument('--topk-k', type=int, default=1)

        parser.add_argument('--random_or_greedy_sampling', action='store_true',default=False)
        parser.add_argument('--random_greedy_ratio', type=float, default=0.5)

        parser.add_argument('--moe-random-sampling', action='store_true', default=False)
        parser.add_argument('--moe-topk-random-sampling', action='store_true', default=False)
        parser.add_argument('--moe-word-level-oracle', action='store_true', default=False)
        parser.add_argument('--num-of-experts', type=int, default=1)
        parser.add_argument('--opposite-expert', action='store_true', default=False)
        parser.add_argument('--decreasing-expert', action='store_true', default=False)

        parser.add_argument('--moe-embedding-random-sampling', action='store_true', default=False)

        parser.add_argument('--random-mix-mle-test', action='store_true', default=False)
        parser.add_argument('--greedy-mix-mle-test', action='store_true', default=False)
        parser.add_argument('--no-mix-gold', action='store_true', default=False)
        parser.add_argument('--replace-non-gold-greedy', action='store_true', default=False)

        parser.add_argument('--random-sampling-greedy-output', action='store_true', default=False)
        parser.add_argument('--random-sampling-random-output', action='store_true', default=False)
        parser.add_argument('--word-oracle-noise-greedy-output', action='store_true', default=False)
        parser.add_argument('--topk-greedy-output', action='store_true', default=False)
        parser.add_argument('--word-oracle-no-noise-greedy-mix-random-output', action='store_true', default=False)
        parser.add_argument('--word-oracle-replace-non-gold-greedy-with-topk', action='store_true', default=False)

        parser.add_argument('--source-random-sampling', action='store_true', default=False)
        parser.add_argument('--source-ss-exponential', type=float, default=0.8)

        parser.add_argument('--iterative-training', action='store_true', default=False)

        parser.add_argument('--soft-training', action='store_true', default=False)
        parser.add_argument('--topk-hard-training', action='store_true', default=False)
        parser.add_argument('--topk-hard', type=int, default=-1)

        parser.add_argument('--switchout', action='store_true', default=False)
        parser.add_argument('--tau-x', type=float, default=1)
        parser.add_argument('--tau-y', type=float, default=1)
        parser.add_argument('--increment-tau', type=float, default=0)
        
        parser.add_argument('--minimum-risk-training', action='store_true', default=False)
        parser.add_argument('--minimum-risk-training-beam-size', type=int, default=100)
        
        parser.add_argument('--decay-update', type=int, default=0)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        generator = None
        if args.use_sentence_level_oracles or args.use_sentence_oracle_mask:
            from fairseq.sequence_generator import SequenceGenerator
            import fairseq.search as search
            search_strategy = search.LengthConstrainedBeamSearch(tgt_dict, min_len_a=1,
                                                                     min_len_b=0,
                                                                     max_len_a=1,
                                                                     max_len_b=0, )
            generator = SequenceGenerator(tgt_dict, beam_size=args.oracle_search_beam_size, match_source_len=False,
                                          max_len_a=1, max_len_b=100, search_strategy=search_strategy)
        elif args.minimum_risk_training:
            from fairseq.sequence_generator import SequenceGenerator
            import fairseq.search as search
            search_strategy = search.Sampling(tgt_dict, sampling_topk=3)
            generator = SequenceGenerator(tgt_dict, beam_size=args.minimum_risk_training_beam_size, search_strategy=search_strategy,
                                          max_len_a= 1, max_len_b=0,
                                           normalize_scores=True)
        
        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder, generator)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def set_num_updates(self, num_updates):
        self.updates = num_updates

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.epoch_decay is True:
            self.decay_prob(epoch, or_type=3, k=self.decay_k)
            print('swith to epoch {}, prob. -> {}'.format(epoch, self.probs))

    def decay_prob(self, i, or_type=4, k=3000, source_decay=False):
        if or_type == 1:  # Linear decay
            or_prob_begin, or_prob_end = 1., 0.
            or_decay_rate = (or_prob_begin - or_prob_end) / 10.
            ss_decay_rate = 0.1
            prob = or_prob_begin - (ss_decay_rate * i)
            if prob < or_prob_end:
                prob_i = or_prob_end
                print('[Linear] schedule sampling probability do not change {}'.format(prob_i))
            else:
                prob_i = prob
                print('[Linear] decay schedule sampling probability to {}'.format(prob_i))

        elif or_type == 2:  # Exponential decay
            prob_i = numpy.power(k, i)
            print('[Exponential] decay schedule sampling probability to {}'.format(prob_i))

        elif or_type == 3:  # Inverse sigmoid decay
            prob_i = k / (k + numpy.exp((i / k)))
            # print('[Inverse] decay schedule sampling probability to {}'.format(prob_i))
        elif or_type == 4:  
            prob_i = math.exp(math.log(self.exponential) * i / k)

        self.probs = prob_i
        return prob_i

    def get_probs(self):
        return self.probs


    def get_word_orcale_tokens(self, pred_logits, prev_output_tokens, epsilon=1e-6):
        B, L = prev_output_tokens.size()
        # B x L x V
        if self.use_greedy_gumbel_noise:
            pred_logits.data.add_(-torch.log(-torch.log(torch.empty_like(
                pred_logits).uniform_(0, 1) + epsilon) + epsilon)) / self.gumbel_noise

        pred_tokens = torch.max(pred_logits, dim=-1)[1]
        bos_idx = prev_output_tokens[0, 0]
        pred_tokens = torch.cat([(bos_idx * torch.ones((B, 1))).to(pred_tokens), pred_tokens], dim=1)[:, :-1]
        sample_gold_prob = self.probs if self.epoch_decay else self.decay_prob(self.updates, k=self.decay_k)
        sample_gold_prob = sample_gold_prob * torch.ones_like(prev_output_tokens, dtype=torch.float32)
        sample_gold_mask = torch.bernoulli(sample_gold_prob).long()

        return prev_output_tokens * sample_gold_mask + pred_tokens * (1 - sample_gold_mask)

    def get_greedy_mix_gold_input_greedy_mix_random_output(self, pred_logits, prev_output_tokens, epsilon=1e-6):
        B, L = prev_output_tokens.size()
        V = len(self.decoder.dictionary.symbols)

        bos_idx = prev_output_tokens[0, 0]
        pred_tokens_no_noise = torch.max(pred_logits, dim=-1)[1]

        # B x L x V
        if self.use_greedy_gumbel_noise:
            pred_logits.data.add_(-torch.log(-torch.log(torch.empty_like(
                pred_logits).uniform_(0, 1) + epsilon) + epsilon)) / self.gumbel_noise
        pred_tokens = torch.max(pred_logits, dim=-1)[1]

        pred_tokens_replace_non_gold = pred_tokens_no_noise.detach().clone()
        random_tokens = torch.randint_like(prev_output_tokens, low=0, high=V)
        non_gold_mask = pred_tokens_replace_non_gold[:, :-1] != prev_output_tokens[:, 1:]
        last_col_mask = torch.zeros(B, 1, device=pred_tokens_replace_non_gold.device).bool()
        non_gold_mask = torch.cat([non_gold_mask, last_col_mask], dim=1)
        #pred_tokens_replace_non_gold.masked_scatter_(non_gold_mask, random_tokens)
        pred_tokens_replace_non_gold.masked_fill_(non_gold_mask, 1)### modify
        pred_tokens_replace_non_gold = torch.cat([(bos_idx * torch.ones((B, 1))).to(pred_tokens_replace_non_gold), pred_tokens_replace_non_gold], dim=1)[:, :-1]

        pred_tokens = torch.cat([(bos_idx * torch.ones((B, 1))).to(pred_tokens), pred_tokens], dim=1)[:, :-1]
        sample_gold_prob = self.probs if self.epoch_decay else self.decay_prob(self.updates, k=self.decay_k)
        sample_gold_prob = sample_gold_prob * torch.ones_like(prev_output_tokens, dtype=torch.float32)
        sample_gold_mask = torch.bernoulli(sample_gold_prob).long()

        return prev_output_tokens * sample_gold_mask + pred_tokens * (1 - sample_gold_mask), pred_tokens_replace_non_gold


    def get_greedy_output(self, pred_logits, prev_output_tokens):
        B,L = prev_output_tokens.size()
        pred_tokens = torch.max(pred_logits, dim=-1)[1]
        bos_idx = prev_output_tokens[0, 0]
        pred_tokens = torch.cat([(bos_idx * torch.ones((B, 1))).to(pred_tokens), pred_tokens], dim=1)[:, :-1]
        return pred_tokens

    def get_topk_output(self, pred_logits, prev_output_tokens):
        B,L = prev_output_tokens.size()
        #p = torch.rand(1)
        #if p>0.5:
        #    pred_tokens = torch.topk(pred_logits, k=2, dim=-1)[1][:,:,1:2].squeeze()
        #else:
        #    pred_tokens = torch.topk(pred_logits, k=2, dim=-1)[1][:,:,0:1].squeeze()
        k=1
        pred_tokens = torch.topk(pred_logits, k=k, dim=-1)[1]

        #assert pred_tokens.dim()==2, pred_tokens.size()
        bos_idx = prev_output_tokens[0, 0]
        pred_tokens = torch.cat([(bos_idx * torch.ones((B, 1, k))).to(pred_tokens), pred_tokens], dim=1)[:, :-1, :]
        return pred_tokens
        


    def get_soft_greedy_output(self, pred_logits):
        pred_prob = torch.nn.functional.softmax(pred_logits, dim=-1)
        return pred_prob



    def get_greedy_noise_input_greedy_output(self, pred_logits, prev_output_tokens, epsilon=1e-6):
        B, L = prev_output_tokens.size()
        bos_idx = prev_output_tokens[0, 0]

       # if not self.soft_training and (not self.topk_hard_training):
        pred_tokens_no_noise = torch.max(pred_logits, dim=-1)[1]
        pred_tokens_no_noise = torch.cat([(bos_idx * torch.ones((B, 1))).to(pred_tokens_no_noise), pred_tokens_no_noise], dim=1)[:, :-1]


        if self.use_greedy_gumbel_noise:
            pred_logits.data.add_(-torch.log(-torch.log(torch.empty_like(
                pred_logits).uniform_(0, 1) + epsilon) + epsilon)) / self.gumbel_noise

        pred_tokens = torch.max(pred_logits, dim=-1)[1]
        pred_tokens = torch.cat([(bos_idx * torch.ones((B, 1))).to(pred_tokens), pred_tokens], dim=1)[:, :-1]

        sample_gold_prob = self.probs if self.epoch_decay else self.decay_prob(self.updates, k=self.decay_k)
        sample_gold_prob = sample_gold_prob * torch.ones_like(prev_output_tokens, dtype=torch.float32)
        sample_gold_mask = torch.bernoulli(sample_gold_prob).long()

        return prev_output_tokens * sample_gold_mask + pred_tokens * (1 - sample_gold_mask), pred_tokens_no_noise

    def get_topk_input_greedy_output(self, pred_logits, prev_output_tokens, k):
        B, L = prev_output_tokens.size()
        topk_tokens = torch.topk(pred_logits, k, dim=-1, largest=True)[1].view(B*L, k)
        random_numbers = torch.randint_like(prev_output_tokens, low=1, high=k).view(-1,1)
        kth_tokens = topk_tokens[torch.arange(B*L, dtype=topk_tokens.dtype, device=topk_tokens.device).view(-1,1), random_numbers].view(B,L)
        pred_tokens = torch.max(pred_logits, dim=-1)[1]

        bos_idx = prev_output_tokens[0, 0]
        topk_pred_tokens = torch.cat([(bos_idx * torch.ones((B, 1), device=kth_tokens.device, dtype=kth_tokens.dtype)), kth_tokens], dim=1)[:, :-1]
        pred_tokens = torch.cat([(bos_idx * torch.ones((B, 1))).to(pred_tokens), pred_tokens], dim=1)[:, :-1]

        sample_gold_prob = self.probs if self.epoch_decay else self.decay_prob(self.updates, k=self.decay_k)
        sample_gold_prob = sample_gold_prob * torch.ones_like(prev_output_tokens, dtype=torch.float32)
        sample_gold_mask = torch.bernoulli(sample_gold_prob).long()

        return prev_output_tokens * sample_gold_mask + topk_pred_tokens * (1 - sample_gold_mask), pred_tokens


    def get_random_tokens(self, prev_output_tokens):
        B, L = prev_output_tokens.size()
        V = len(self.decoder.dictionary.symbols)
        random_tokens = torch.randint_like(prev_output_tokens, low=0, high=V)
        bos_idx = prev_output_tokens[0, 0]
        random_tokens = torch.cat([(bos_idx * torch.ones((B, 1))).to(random_tokens), random_tokens], dim=1)[:, :-1]
        sample_gold_prob = self.probs if self.epoch_decay else self.decay_prob(self.updates, k=self.decay_k)
        sample_gold_prob = sample_gold_prob * torch.ones_like(prev_output_tokens, dtype=torch.float32)
        sample_gold_mask = torch.bernoulli(sample_gold_prob).long()

        return prev_output_tokens * sample_gold_mask + random_tokens * (1 - sample_gold_mask)
        

        
    def replace_non_gold_greedy_with_random_tokens(self, pred_logits, prev_output_tokens):
        B, L = prev_output_tokens.size()
        B1, L1, V1 =  pred_logits.size()
        V = len(self.decoder.dictionary.symbols)
        assert V == V1
        assert B == B1
        assert L == L1
        random_tokens = torch.randint_like(prev_output_tokens, low=0, high=V)
        pred_tokens = torch.max(pred_logits, dim=-1)[1]

        non_gold_mask = pred_tokens[:,:-1] != prev_output_tokens[:,1:]
        last_col_mask = torch.zeros(B, 1, device=pred_tokens.device).bool()
        non_gold_mask = torch.cat([non_gold_mask, last_col_mask], dim=1)

        pred_tokens.masked_scatter_(non_gold_mask, random_tokens)
        bos_idx = prev_output_tokens[0, 0]
        pred_tokens = torch.cat([(bos_idx * torch.ones((B, 1))).to(pred_tokens), pred_tokens], dim=1)[:, :-1]

        return pred_tokens



    def get_topk_tokens(self, pred_logits, prev_output_tokens, k):
        B, L = prev_output_tokens.size()
        topk_tokens = torch.topk(pred_logits, k, dim=-1, largest=True)[1].view(B*L, k)
        random_numbers = torch.randint_like(prev_output_tokens, low=1, high=k).view(-1,1)
        kth_tokens = topk_tokens[torch.arange(B*L, dtype=topk_tokens.dtype, device=topk_tokens.device).view(-1,1), random_numbers].view(B,L)

        bos_idx = prev_output_tokens[0, 0]
        pred_tokens = torch.cat([(bos_idx * torch.ones((B, 1), device=kth_tokens.device, dtype=kth_tokens.dtype)), kth_tokens], dim=1)[:, :-1]
        sample_gold_prob = self.probs if self.epoch_decay else self.decay_prob(self.updates, k=self.decay_k)
        sample_gold_prob = sample_gold_prob * torch.ones_like(prev_output_tokens, dtype=torch.float32)
        sample_gold_mask = torch.bernoulli(sample_gold_prob).long()

        return prev_output_tokens * sample_gold_mask + pred_tokens * (1 - sample_gold_mask)


    def get_random_or_greedy_tokens(self, pred_logits, prev_output_tokens, epsilon=1e-6):
        B, L = prev_output_tokens.size()
        V = pred_logits.size(-1)
        # B x L x V
        if self.use_greedy_gumbel_noise:
            pred_logits.data.add_(-torch.log(-torch.log(torch.Tensor(
                pred_logits.size()).to(pred_logits).uniform_(0, 1) + epsilon) + epsilon)) / self.gumbel_noise

        greedy_tokens = torch.max(pred_logits, dim=-1)[1]
        random_tokens = torch.randint_like(prev_output_tokens, low=0, high=V)

        sample_random_greedy_prob = self.random_greedy_ratio * torch.ones_like(prev_output_tokens, dtype=torch.float32)
        sample_random_greedy_mask = torch.bernoulli(sample_random_greedy_prob).long()
        pred_tokens = greedy_tokens * sample_random_greedy_mask + random_tokens * (1 - sample_random_greedy_mask)

        bos_idx = prev_output_tokens[0, 0]
        pred_tokens = torch.cat([(bos_idx * torch.ones((B, 1))).to(pred_tokens), pred_tokens], dim=1)[:, :-1]
        sample_gold_prob = self.decay_prob(self.epoch-5, or_type=3, k=self.decay_k) if self.epoch_decay else self.decay_prob(self.updates, k=self.decay_k)
        sample_gold_prob = sample_gold_prob * torch.ones_like(prev_output_tokens, dtype=torch.float32)
        sample_gold_mask = torch.bernoulli(sample_gold_prob).long()

        return prev_output_tokens * sample_gold_mask + pred_tokens * (1 - sample_gold_mask)
    
    
    def compute_sentence_bleu(self, pred, ref):
        pre_str = [str(i) for i in pred]
        ref_str = [str(i) for i in ref]
        bleu = nltk.translate.bleu_score.sentence_bleu([ref_str], pre_str)
        return bleu
        
    @torch.no_grad()
    def get_sentence_oracle_tokens(self, prev_output_tokens, src_tokens, src_lengths, target):
        bos_idx = prev_output_tokens[0, 0]
        B, L = prev_output_tokens.size()
        sample = {}
        sample['net_input'] = {}
        sample['net_input']['src_tokens'] = src_tokens
        sample['net_input']['src_lengths'] = src_lengths
        noise = None
        if self.use_bleu_gumbel_noise:
            noise = self.gumbel_noise
        out = self.generator.generate([self], sample, target, noise=noise)
        sentence_oracle_inputs = torch.ones_like(target)
        
        i = 0
        for x, t in zip(out, target):
            tmp_max = 0
            best_item = None
            for item in x:
                bleu = self.compute_sentence_bleu(item['tokens'].cpu().numpy().tolist(), t.cpu().numpy().tolist())
                
                if bleu > tmp_max:
                    best_item = item
                    tmp_max = bleu
            sentence_oracle_inputs[i, :len(best_item['tokens']) - 1] = best_item['tokens'][:-1]
            i += 1
        oracle_inputs = torch.cat([bos_idx * torch.ones((B, 1), device=prev_output_tokens.device, dtype=torch.int64),
                                   sentence_oracle_inputs], dim=1)[:, :-1]
        self.train()
        return oracle_inputs
        
    @torch.no_grad()
    def get_sentence_oracle_tokens_mask(self, prev_output_tokens, src_tokens, src_lengths, target):
        prob = self.probs if self.epoch_decay else self.decay_prob(self.updates, or_type=4, k=self.decay_update)
        bos_idx = prev_output_tokens[0, 0]
        B, L = prev_output_tokens.size()
        sample = {}
        sample['net_input'] = {}
        sample['net_input']['src_tokens'] = src_tokens
        sample['net_input']['src_lengths'] = src_lengths
        noise = None
        if self.use_bleu_gumbel_noise:
            noise = self.gumbel_noise
        out = self.generator.generate([self], sample, target, noise=noise)
        sentence_oracle_inputs = torch.ones_like(target)
        
        mask_prob = prob * torch.ones_like(prev_output_tokens, dtype=torch.float32)
        mask = torch.bernoulli(mask_prob).long()
        
        i = 0
        for x, t in zip(out, target):
            tmp_max = 0
            best_item = None
            for item in x:
                bleu = self.compute_sentence_bleu(item['tokens'].cpu().numpy().tolist(), t.cpu().numpy().tolist())
                
                if bleu > tmp_max:
                    best_item = item
                    tmp_max = bleu
            sentence_oracle_inputs[i, :len(best_item['tokens']) - 1] = best_item['tokens'][:-1]
            i += 1
        oracle_inputs = torch.cat([bos_idx * torch.ones((B, 1), device=prev_output_tokens.device, dtype=torch.int64),
                                   sentence_oracle_inputs], dim=1)[:, :-1]
        self.train()
        return prev_output_tokens * mask + oracle_inputs * (1-mask)
            
    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            target=None,
            cls_input: Optional[Tensor] = None,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, cls_input=cls_input,
                                   return_all_hiddens=return_all_hiddens, )

        with torch.no_grad():
            if self.training and self.use_word_oracle:
                if self.epoch > 5:
                    decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, features_only=features_only,
                                               src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, )
                    prev_output_tokens = self.get_word_orcale_tokens(decoder_out[0].detach(), prev_output_tokens)
                
            
            elif self.training and self.use_sentence_oracle:
                prob = self.probs if self.epoch_decay else self.decay_prob(self.updates, or_type=4, k=self.decay_update)
                p = random.random()
                if p > prob:
                    prev_output_tokens = self.get_sentence_oracle_tokens(
                            prev_output_tokens, src_tokens, src_lengths, target)
                            
            elif self.training and self.use_sentence_oracle_mask:
                prev_output_tokens = self.get_sentence_oracle_tokens_mask(
                            prev_output_tokens, src_tokens, src_lengths, target)            
            


            elif self.training and self.topk_sampling:
                if self.epoch > 5:
                    decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, features_only=features_only,
                                               src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, )
                    prev_output_tokens = self.get_topk_tokens(decoder_out[0].detach(), prev_output_tokens, self.topk_k)

            elif self.training and self.random_or_greedy_sampling:
                if self.epoch > 5:
                    decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, features_only=features_only,
                                               src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, )
                    prev_output_tokens = self.get_random_or_greedy_tokens(decoder_out[0].detach(), prev_output_tokens)



            elif self.training and self.random_sampling_random_output and self.epoch > 5:
                decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, features_only=features_only,
                                           src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, )
                prev_output_tokens, greedy_output = self.get_random_input_random_output_replace_nongold(decoder_out[0].detach(), prev_output_tokens)

            elif self.training and self.word_oracle_noise_greedy_output and self.epoch > 5:
                decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, features_only=features_only,
                                           src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, )
                prev_output_tokens, greedy_output = self.get_greedy_noise_input_greedy_output(decoder_out[0].detach(), prev_output_tokens)
                

            elif self.training and self.topk_greedy_output and self.epoch > 5:
                decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, features_only=features_only,
                                           src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, )
                prev_output_tokens, greedy_output = self.get_topk_input_greedy_output(decoder_out[0].detach(), prev_output_tokens, self.topk_k)

            elif self.training and self.word_oracle_no_noise_greedy_mix_random_output and self.epoch > 5:
                decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, features_only=features_only,
                                           src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, )
                prev_output_tokens, greedy_mix_random_output = self.get_greedy_mix_gold_input_greedy_mix_random_output(decoder_out[0].detach(), prev_output_tokens)

            elif self.training and self.word_oracle_replace_non_gold_greedy_with_topk and self.epoch > 5:
                decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, features_only=features_only,
                                           src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, )
                prev_output_tokens, gold_mix_topk_output = self.get_gold_mix_greedy_input_gold_mix_topk_output(decoder_out[0].detach(), prev_output_tokens)
                

        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, features_only=features_only,
                                       src_lengths=src_lengths, return_all_hiddens=return_all_hiddens)



        if self.training and self.epoch > 5 and self.random_mix_mle_test:
            if self.replace_non_gold_greedy:
                prev_output_tokens = self.replace_non_gold_greedy_with_random_tokens(decoder_out[0].detach(), prev_output_tokens)
            else:
                prev_output_tokens = self.get_random_tokens(prev_output_tokens)
        if self.training and self.epoch > 5 and self.greedy_mix_mle_test:
            if self.no_mix_gold:
                #if not self.soft_training:
                prev_output_tokens = self.get_greedy_output(decoder_out[0].detach(), prev_output_tokens)
                #else:
                #prev_output_tokens = self.get_soft_greedy_output(decoder_out[0].detach())
            else:
                prev_output_tokens = self.get_word_orcale_tokens(decoder_out[0].detach(), prev_output_tokens)



        if self.training and self.random_sampling_greedy_output and self.epoch > 5:
            prev_output_tokens = greedy_output
        elif self.training and self.random_sampling_random_output and self.epoch > 5:
            prev_output_tokens = greedy_output
        elif self.training and self.word_oracle_noise_greedy_output and self.epoch > 5:
            prev_output_tokens = greedy_output
        elif self.training and self.use_sentence_oracle:
            prev_output_tokens = prev_output_tokens
        elif self.training and self.use_sentence_oracle_mask:
            prev_output_tokens = prev_output_tokens
        elif self.training and self.topk_greedy_output and self.epoch > 5:
            prev_output_tokens = greedy_output
        elif self.training and self.word_oracle_no_noise_greedy_mix_random_output and self.epoch > 5:
            prev_output_tokens = greedy_mix_random_output
        elif self.training and self.word_oracle_replace_non_gold_greedy_with_topk and self.epoch > 5:
            prev_output_tokens = gold_mix_topk_output


        if self.use_neighbor_MLE:
            out = [decoder_out, prev_output_tokens, self.updates, self.decay_k]
        else:
            out = decoder_out
        return out


    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Dict[str, List[Optional[Tensor]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout = args.dropout
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.layer_wise_attention = getattr(args, "layer_wise_attention", False)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def build_encoder_layer(self, args):
        return TransformerEncoderLayer(args)

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(
            self,
            src_tokens,
            src_lengths,
            cls_input: Optional[Tensor] = None,
            return_all_hiddens: bool = False,
    ):
        if self.layer_wise_attention:
            return_all_hiddens = True

        x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.empty(1).uniform_()
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x
                
        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_encoder_out: Dict[str, Tensor] = {}

        new_encoder_out["encoder_out"] = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_out["encoder_padding_mask"] = (
            encoder_out.encoder_padding_mask
            if encoder_out.encoder_padding_mask is None
            else encoder_out.encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_out["encoder_embedding"] = (
            encoder_out.encoder_embedding
            if encoder_out.encoder_embedding is None
            else encoder_out.encoder_embedding.index_select(0, new_order)
        )

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out["encoder_out"],  # T x B x C
            encoder_padding_mask=new_encoder_out["encoder_padding_mask"],  # B x T
            encoder_embedding=new_encoder_out["encoder_embedding"],  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
                not hasattr(self, "_future_mask")
                or self._future_mask is None
                or self._future_mask.device != tensor.device
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(
                    utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
                )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout = args.dropout
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.cross_self_attention = getattr(args, "cross_self_attention", False)
        self.layer_wise_attention = getattr(args, "layer_wise_attention", False)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        self.adaptive_softmax = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(
                args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return TransformerDecoderLayer(args, no_encoder_attn)

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
            **kwargs
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        num_expert = kwargs.get('num_expert',1)
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            num_expert=num_expert
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            **kwargs
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            encoder_state: Optional[Tensor] = None
            if encoder_out is not None:
                if self.layer_wise_attention:
                    encoder_states = encoder_out.encoder_states
                    assert encoder_states is not None
                    encoder_state = encoder_states[idx]
                else:
                    encoder_state = encoder_out.encoder_out

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.empty(1).uniform_()
            if not self.training or (dropout_probability > self.decoder_layerdrop):
                x, layer_attn, _ = layer(
                    x,
                    encoder_state,
                    encoder_out.encoder_padding_mask
                    if encoder_out is not None
                    else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                )
                inner_states.append(x)
                if layer_attn is not None and idx == alignment_layer:
                    attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
                self._future_mask.size(0) == 0
                or (not self._future_mask.device == tensor.device)
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    # Overwirte the method to temporaily soppurt jit scriptable in Transformer
    @torch.jit.export
    def reorder_incremental_state(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            new_order: Tensor,
    ):
        """Scriptable reorder incremental state in the transformer."""
        for layer in self.layers:
            layer.reorder_incremental_state(incremental_state, new_order)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("oracle_transformer", "oracle_transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)


@register_model_architecture("oracle_transformer", "oracle_transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


@register_model_architecture("oracle_transformer", "oracle_transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("oracle_transformer", "oracle_transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


@register_model_architecture("oracle_transformer", "oracle_transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, "dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("oracle_transformer", "oracle_transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture("oracle_transformer", "oracle_transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)
