# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion



def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True, **kwargs):
    use_mix_CE = kwargs.get('use_mix_CE', False)

    neighbors = kwargs.get('neighbors', None)
    cur_num_updates = kwargs.get('cur_num_updates', None)
    total_num_updates = kwargs.get('total_num_updates', None)

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
        if neighbors is not None and neighbors.dim() == lprobs.dim() - 1 :
            neighbors = neighbors.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)
    
    if use_mix_CE:
        p = (cur_num_updates) / (total_num_updates * 2)
        mix_input_neighbor_loss = -lprobs.gather(dim=-1, index=neighbors)
        nll_loss = (1 - p) * nll_loss + p * mix_input_neighbor_loss

    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('oracle_label_smoothed_cross_entropy')
class OracleLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, distributed_world_size):
        super().__init__(task)
        self.eps = label_smoothing
        self.sentence_avg = sentence_avg
        self.GPU_nums = distributed_world_size

        self.mix_margin_nll_loss = torch.nn.MarginRankingLoss(margin=0.05)                                      ###

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        use_mix_CE = False
        neighbors = None
        current_num_updates = None
        total_num_updates = None

        net_output = model(**sample['net_input'], target=sample['target'])


        if isinstance(net_output, list) and len(net_output)==4:
            use_mix_CE = True
            total_num_updates = net_output[3]
            current_num_updates = net_output[2]
            neighbors = net_output[1]
            net_output = net_output[0]

        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce,
                                                   neighbors=neighbors,
                                                   total_num_updates=total_num_updates,
                                                   cur_num_updates=current_num_updates,
                                                   use_mix_CE=use_mix_CE)

        prob = model.get_probs()
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'prob': prob,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'gpu_nums': 1,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, 
                                                      neighbors = None,
                                                      total_num_updates = None,
                                                      cur_num_updates = None,
                                                      use_mix_CE = False,
                                                      **kwargs):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)

        if neighbors is not None:
            B, L = neighbors.size()
            bos = neighbors[:, 0]
            neighbors = torch.cat([neighbors, bos.unsqueeze(1)], dim=1)[:,1:]
            assert neighbors.size(0) == B and neighbors.size(1) == L
            neighbors = neighbors.contiguous().view(-1, 1)

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            neighbors=neighbors, 
            cur_num_updates=cur_num_updates,
            total_num_updates=total_num_updates,
            use_mix_CE=use_mix_CE
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        GPU_nums = sum(log.get('gpu_nums', 0) for log in logging_outputs)
        prob = sum(log.get('prob', 0) for log in logging_outputs) / GPU_nums
        metrics.log_scalar('prob', prob)
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
