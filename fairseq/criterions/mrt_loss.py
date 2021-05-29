# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
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


@register_criterion('MRT_loss')
class MRTLoss(FairseqCriterion):
    def __init__(self, task, sentence_avg, label_smoothing, distributed_world_size):
        super().__init__(task)
        self.eps = label_smoothing
        self.sentence_avg = sentence_avg
        self.GPU_nums = distributed_world_size
    
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')    
    
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample['net_input'], target=sample['target'])
        if len(net_output) == 4:
            batch_bleus, batch_outputs, decoder_out, shape = net_output[0], net_output[1], net_output[2], net_output[3]
            loss,nll_loss = self.compute_loss(model, batch_bleus, batch_outputs, decoder_out, shape, reduce=reduce)
        else:
            loss, nll_loss = self.compute_CE_loss(model, net_output, sample, reduce=True)

        #prob = model.get_probs()
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
           # 'prob': prob,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'gpu_nums': 1,
        }
        return loss, sample_size, logging_output
    
    def compute_CE_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    def compute_loss(self, model, batch_bleus, batch_outputs, decoder_out, shape, reduce=True):
        try:
            lprobs = model.get_normalized_probs(decoder_out, log_probs=True)
            #print(lprobs.size())
            lprobs = lprobs.view(-1, lprobs.size(-1))
            batch_outputs = batch_outputs.view(-1, 1)
            nll = lprobs.gather(dim=-1, index=batch_outputs)
            pad_mask = batch_outputs.eq(self.padding_idx)

            nll.masked_fill_(pad_mask, 0.)
            #if shape[0]>4:
            #    print(batch_outputs.view(shape)[1])
            #    print(torch.sum(batch_outputs.view(shape) != 1, dim=-1))
            seq_nll = nll.view(shape).sum(dim=-1)
            neg_batch_bleus = -1 * torch.tensor(batch_bleus, device=seq_nll.device, dtype=torch.float)

            #print('seq_nll:', seq_nll.size(), lprobs.size(),shape)
            #print(torch.max(seq_nll, dim=-1)[0], torch.min(seq_nll,dim=-1)[0])
            #print(seq_nll)
            seq_nll = seq_nll-torch.max(seq_nll, dim=-1)[0].unsqueeze(1)
            seq_probs = torch.exp(seq_nll*0.005)
            normalizer = torch.sum(seq_probs, dim=-1).view(-1, 1)
            normalized_probs = seq_probs.div(normalizer)
            #print('norm prob:', torch.isfinite(normalized_probs).all(), torch.isfinite(neg_batch_bleus).all())
            #print(normalizer)
            loss = (normalized_probs * neg_batch_bleus).sum()
            #print('==0', torch.all(normalizer.eq(0)), torch.max(normalizer), torch.min(normalizer), torch.eq(torch.min(normalizer),0), torch.max(seq_probs))
            if torch.min(normalizer) == 0:
                print(torch.min(normalizer, dim=-1))
            #print('lprob:', torch.any(torch.isinf(lprobs)))
            #print('nll:', torch.any(torch.isinf(nll)))
            #print('seq_nll:', torch.any(torch.isinf(seq_nll)))
            #print('seq_probs:', torch.any(torch.isinf(seq_probs)))
            #if torch.isnan(loss):
            #    print('pad_mask:', pad_mask.size())
                #torch.save( pad_mask,'pad_mask.pt')
            #print('$$$$$$$$$$$$$$$$$$$$$$')
            nll_loss = loss
            return loss, nll_loss

        except:
            print()




        
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
