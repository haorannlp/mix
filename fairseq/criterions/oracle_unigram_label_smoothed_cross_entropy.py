# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True, **kwargs):
    use_neighbor_MLE = kwargs.get('use_neighbor_MLE', False)
    super_neighbor_MLE = kwargs.get('super_neighbor_MLE', False)
    greedy_out_in_2pass = kwargs.get('greedy_out_in_2pass')
    assert (use_neighbor_MLE + super_neighbor_MLE) <=1
    dynamic_mix_mle = True


    neighbors = kwargs.get('neighbors', None)
    cur_num_updates = kwargs.get('cur_num_updates', -1)
    total_num_updates = kwargs.get('total_num_updates', 0)

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
        if neighbors is not None and neighbors.dim() == lprobs.dim() - 1 :
            neighbors = neighbors.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)
        
    # dual skew
    # if cur_num_updates > 30000:
    #   probs = torch.nn.functional.softmax(lprobs, dim=-1)
    #    weighted_target = 0.99*target.float()
    #    p1 = -probs.gather(dim=-1,index=target) * lprobs.gather(dim=-1, index=target)
    #    p2 = probs.gather(dim=-1,index=target) * torch.log(0.99+0.01*probs.gather(dim=-1, index=target))
    #    nll_loss = -(p1+p2)
    
    if use_neighbor_MLE:
        #p = 0.6*(cur_num_updates) / (total_num_updates * 2)
        #p = 0.5
        # p = 0.6*torch.pow(10, torch.FloatTensor([cur_num_updates/total_num_updates -1]))
        p = (cur_num_updates) / (total_num_updates * 2)
        if neighbors.size(-1) == 1:
            mix_input_neighbor_loss = -lprobs.gather(dim=-1, index=neighbors)
            nll_loss = (1 - p) * nll_loss + p * mix_input_neighbor_loss
        elif neighbors.size(-1) != 1:
            #raise ValueError
            #assert lprobs.size() == neighbors.size(), (lprobs.size(), neighbors.size())
            #neighbors = neighbors.to(lprobs.dtype)
            #mix_input_neighbor_loss = -torch.sum(lprobs * neighbors, dim=1)
            BL, K = neighbors.size()
            mix_neighbor_losses = torch.zeros(BL, K).to(lprobs)
            def get_ps(p, k):
                temp = [(1-p) * p**i for i in range(0,k)]
                temp.extend([p**k])
                return torch.FloatTensor(temp).to(lprobs).unsqueeze(1)
            coef = get_ps(p, K)
            assert coef.dim() == 2
            assert coef.size(0) == K+1
            assert coef.size(1) == 1
            
            for j in range(K):
                mix_neighbor_losses[:, j:j+1] = -lprobs.gather(dim=-1, index=neighbors[:, j:j+1])
                
            nll_loss = torch.cat([nll_loss, mix_neighbor_losses], dim=1)
            
            assert nll_loss.dim() == 2
            assert nll_loss.size(0) == BL
            assert nll_loss.size(1) == (K+1)
            nll_loss = nll_loss.mm(coef)
            
        
        
#    elif super_neighbor_MLE:
#        p = (cur_num_updates) / (total_num_updates * 2)
#        if neighbors.size(-1) == 1:
#            mix_input_neighbor_loss = -lprobs.gather(dim=-1, index=neighbors)
#            mix_input_greedy_out_2_pass_loss = -lprobs.gather(dim=-1, index=greedy_out_in_2pass)
#        elif neighbors.size(-1) != 1:
#            raise ValueError
##        #nll_loss = (1 - p) * nll_loss + p * (mix_input_neighbor_loss+mix_input_greedy_out_2_pass_loss) / 2
#        nll_loss = (1 - p) * nll_loss + p * (mix_input_greedy_out_2_pass_loss) 

    # if dynamic_mix_mle:
    #     p = (cur_num_updates) / (total_num_updates * 2)
    #     probs = torch.nn.functional.softmax(lprobs.detach().clone(), dim=-1)          # no need gradient
    #     coe_neighbor = probs.gather(dim=-1, index=neighbors)
    #     coe_neighbor = coe_neighbor / (p + coe_neighbor)
    #     mix_input_neighbor_loss = -lprobs.gather(dim=-1, index=neighbors)
    #     nll_loss = (1 - coe_neighbor) * nll_loss + coe_neighbor * mix_input_neighbor_loss


    # if (neighbors is not None) and (cur_num_updates >= 0):
    #     # One pass MIMO x5
    #     p = (cur_num_updates) / (total_num_updates*2)
    #     mix_input_neighbor_loss = -lprobs.gather(dim=-1, index=neighbors)
    #     nll_loss = (1-p) * nll_loss + p * mix_input_neighbor_loss

    ### TODO: train without smooth loss ?
    random_targets = torch.randint_like(target, low=3, high=lprobs.size(-1))
#    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    smooth_loss = -lprobs.gather(dim=-1, index=random_targets)
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
    #eps_i = epsilon / lprobs.size(-1)
    eps_i = epsilon 
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('oracle_unigram_label_smoothed_cross_entropy')
class OracleUnigramLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

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

        use_neighbor_MLE = False
        super_neighbor_MLE = False

        net_output = model(**sample['net_input'], target=sample['target'])

        # neighbors = net_output[1]
        # use_gold = net_output[2]
        # cur_num_updates = net_output[3]
        # total_num_updates = net_output[4]
        # greedy_mix_mle = net_output[5]
        # use_neighbor_MLE = net_output[6]
        # net_output = net_output[0]
        #
        # loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce,
        #                                    neighbors=neighbors,
        #                                    cur_num_updates=cur_num_updates,
        #                                    total_num_updates=total_num_updates,
        #                                    use_gold=use_gold,
        #                                    greedy_mix_mle=greedy_mix_mle,
        #                                    use_neighbor_MLE=use_neighbor_MLE)

        #为iterative training 让路
        if isinstance(net_output, list) and len(net_output)==4:        ###
            use_neighbor_MLE = True             ###
            total_num_updates = net_output[3]###
            current_num_updates = net_output[2]  ###
            neighbors = net_output[1]           ###
            net_output = net_output[0]          ###
        elif isinstance(net_output, list) and len(net_output)==5:        ###
            super_neighbor_MLE = True             ###
            greedy_out_in_2pass = net_output[4]
            total_num_updates = net_output[3]###
            current_num_updates = net_output[2]  ###
            neighbors = net_output[1]           ###
            net_output = net_output[0]          ###

        if use_neighbor_MLE:
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce,
                                                   neighbors=neighbors,
                                                   total_num_updates=total_num_updates,
                                                   cur_num_updates=current_num_updates,
                                                   use_neighbor_MLE=use_neighbor_MLE)
        elif super_neighbor_MLE:
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce,
                                                   neighbors=neighbors,
                                                   total_num_updates=total_num_updates,
                                                   cur_num_updates=current_num_updates,
                                                   super_neighbor_MLE=super_neighbor_MLE,
                                                   greedy_out_in_2pass=greedy_out_in_2pass)
        else:
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce,
                                                   use_neighbor_MLE=use_neighbor_MLE)

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

    def compute_loss(self, model, net_output, sample, reduce=True, **kwargs):
        use_neighbor_MLE = kwargs.get('use_neighbor_MLE',False)
        super_neighbor_MLE = kwargs.get('super_neighbor_MLE',False)
        total_num_updates = kwargs.get('total_num_updates', None)
        cur_num_updates = kwargs.get('cur_num_updates', -1)
        
        neighbors = kwargs.get('neighbors', None)
        greedy_out_in_2pass = kwargs.get('greedy_out_in_2pass',None)
        
        use_gold = kwargs.get('use_gold', True)
        greedy_mix_mle = kwargs.get('greedy_mix_mle', False)

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)


        if neighbors is not None and neighbors.dim()==2:
            B, L = neighbors.size()
            bos = neighbors[:, 0]
            neighbors = torch.cat([neighbors, bos.unsqueeze(1)], dim=1)[:,1:]
            assert neighbors.size(0) == B and neighbors.size(1) == L
            neighbors = neighbors.contiguous().view(-1, 1)
        elif neighbors is not None and neighbors.dim()==3:
            bos = neighbors[:, 0, :]
            neighbors = torch.cat([neighbors, bos.unsqueeze(1)], dim=1)[:,1:,:]
            neighbors = neighbors.contiguous().view(-1, neighbors.size(-1))
            
        if greedy_out_in_2pass is not None and greedy_out_in_2pass.dim()==2:
            B, L = greedy_out_in_2pass.size()
            bos = greedy_out_in_2pass[:, 0]
            greedy_out_in_2pass = torch.cat([greedy_out_in_2pass, bos.unsqueeze(1)], dim=1)[:,1:]
            assert greedy_out_in_2pass.size(0) == B and greedy_out_in_2pass.size(1) == L
            greedy_out_in_2pass = greedy_out_in_2pass.contiguous().view(-1, 1)

        # loss, nll_loss = pseudo_loss(
        #     lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        #     neighbors=neighbors, cur_num_updates=cur_num_updates,
        #     total_num_updates=total_num_updates,
        #     use_neighbor_MLE=use_neighbor_MLE)

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            neighbors=neighbors, cur_num_updates=cur_num_updates,
            total_num_updates=total_num_updates,
            use_neighbor_MLE=use_neighbor_MLE,
            super_neighbor_MLE=super_neighbor_MLE,
            greedy_out_in_2pass=greedy_out_in_2pass
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

