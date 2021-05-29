# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import operator
import torch
import nltk, re

from fairseq import metrics, utils, bleu
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data import data_utils

from fairseq.sequence_generator import SequenceGenerator
import fairseq.search as search
import torch.nn.functional as F


class BleuScorer(object):

    def __init__(self, pad, eos, unk):
        self._scorer = bleu.Scorer(pad, eos, unk)

    def score(self, ref, hypo):
        self._scorer.reset(one_init=True)
        self._scorer.add(ref, hypo)
        return self._scorer.score()

SPACE_NORMALIZER = re.compile(r"\s+")
def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()

@register_criterion('seq_nll_loss')
class SequenceNLLCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.dst_dict = task.target_dictionary
        self.pad = self.dst_dict.pad()
        self.eos = self.dst_dict.eos()
        self.unk = self.dst_dict.unk()

        search_strategy = search.LengthConstrainedBeamSearch(task.target_dictionary, min_len_a=0,
                                                             min_len_b=1,
                                                             max_len_a=1.2,
                                                             max_len_b=5, )
        self._generator = SequenceGenerator(task.target_dictionary, beam_size=5, match_source_len=False,
                                      max_len_a=1.2, max_len_b=5, search_strategy=search_strategy)
        self._scorer = BleuScorer(self.pad, self.eos, self.unk)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def _generate_hypotheses(self, model, sample):
        hypos = self._generator.generate([model], sample)

        return hypos

    def compute_sentence_bleu(self, ref, pred):
        pre_str = [str(i) for i in pred]
        ref_str = [str(i) for i in ref]
        bleu = nltk.translate.bleu_score.sentence_bleu([ref_str], pre_str,smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method2)
        return bleu

    def tokenize(self, line, dict, tokenize=tokenize_line, add_if_not_exist=True, consumer=None):
        words = tokenize(line)
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1)
        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = dict.add_symbol(word)
            else:
                idx = dict.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        ids[nwords] = dict.eos_index
        return ids

    def add_bleu_to_hypotheses(self, sample, hypos):
        if 'include_bleu' in sample:
            return hypos

        sample['include_bleu'] = True

        target = sample['target'].data.int()
        for i, hypos_i in enumerate(hypos):
            ref = utils.strip_pad(target[i, :], self.pad).cpu()  # 加了numpy tolist
            r = self.dst_dict.string(ref, bpe_symbol='@@ ', escape_unk=True)
            r = self.tokenize(r, self.dst_dict, add_if_not_exist=True)
            # don't need to detokenize and tokenize
            for j, hypo in enumerate(hypos_i):
                h = self.dst_dict.string(hypo['tokens'].int().cpu(), bpe_symbol='@@ ')
                h = self.tokenize(h, self.dst_dict, add_if_not_exist=True)
                hypo['bleu'] = self._scorer.score(r, h)
                print('index:', j, ' len:', len(r), len(h), ' bleu:',  hypo['bleu'])
                #self.compute_sentence_bleu(ref, hypo['tokens'].cpu().numpy().tolist())
                #print('bleu:', self.compute_sentence_bleu(ref, hypo['tokens'].cpu().numpy().tolist()), ref, hypo['tokens'].cpu().numpy().tolist())
                #print('bleu:', self.compute_sentence_bleu(ref, hypo['tokens'].cpu().numpy().tolist()))
        return hypos

    def prepare_sample_and_hypotheses(self, model, sample, hypos):
        """Apply criterion-specific modifications to the given sample/hypotheses."""
        # compute BLEU for each hypothesis
        hypos = self.add_bleu_to_hypotheses(sample, hypos)
        # for each sentence, find the hypothesis with the max BLEU and set it as the "target" hypothsis
        max_index = [
            max(enumerate(x['bleu'] for x in h), key=operator.itemgetter(1))[0]
            for h in hypos
        ]
        sample['target_hypo_idx'] = sample['target'].data.new(max_index)
        return sample, hypos

    def _update_sample_with_hypos(self, sample, hypos):
        num_hypos_per_batch = len(hypos[0])
        assert all(len(h) == num_hypos_per_batch for h in hypos)

        def repeat_num_hypos_times(t, dim=1):
            return t.repeat(1, num_hypos_per_batch).view(num_hypos_per_batch*t.size(0), t.size(1)) if dim==1 else \
                t.repeat(num_hypos_per_batch).view(num_hypos_per_batch*t.size(0))

        bsz = sample['net_input']['src_tokens'].size(0)
        sample['net_input']['src_tokens'].data = repeat_num_hypos_times(sample['net_input']['src_tokens'].data)
        sample['net_input']['src_lengths'].data = repeat_num_hypos_times(sample['net_input']['src_lengths'].data, dim=0)

        input_hypos = [h['tokens'] for hypo_i in hypos for h in hypo_i]
        sample['hypotheses'] = data_utils.collate_tokens(
            input_hypos, self.pad, self.eos, left_pad=False, move_eos_to_beginning=False)
        sample['net_input']['prev_output_tokens'] = data_utils.collate_tokens(
            input_hypos, self.pad, self.eos, left_pad=False, move_eos_to_beginning=True)

        sample['target'].data = repeat_num_hypos_times(sample['target'].data)
        sample['ntokens'] = sample['target'].data.ne(self.pad).sum()
        sample['nsentences'] = bsz
        sample['num_hypos_per_batch'] = num_hypos_per_batch
        return sample

    def get_hypothesis_scores(self, net_output, sample, score_pad=False):
        '''return a tensor of model scores for each hypothesis.
        The returned tensor has dimensions [bsz, nhypos, hypolen].
        '''
        bsz, nhypos, hypolen, _ = net_output.size()
        hypotheses = sample['hypotheses'].view(bsz, nhypos, hypolen, 1)
        scores = net_output.gather(3, hypotheses)
        if not score_pad:
            scores = scores * hypotheses.ne(self.pad).float()

        return scores.squeeze(3)

    def get_hypothesis_lengths(self, net_output, sample):
        '''return a tensor of hypothesis lengths
        The returned tensor has dimensions [bsz, nhypos].
        '''
        bsz, nhypos, hypolen, _ = net_output.size()
        lengths = sample['hypotheses'].view(bsz, nhypos, hypolen).ne(self.pad).sum(2).float()
        return lengths


    def sequence_forward(self, net_output, model, sample):
        scores = self.get_hypothesis_scores(net_output, sample)
        lengths = self.get_hypothesis_lengths(net_output, sample)
        avg_scores = scores.sum(2) / lengths
        loss = F.cross_entropy(avg_scores, sample['target_hypo_idx'], size_average=False)
        if torch.any(torch.isinf(loss)) or torch.any(torch.isinf(avg_scores)):
            print(loss)
        if torch.any(torch.isnan(loss)) or torch.any(torch.isnan(avg_scores)):
            print(loss)
        sample_size = net_output.size(0) #sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        print('sample_size:', sample_size)
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        mdoel_state = model.training
        if mdoel_state:
            model.train(False)

        # generate hypotheses
        hypos = self._generate_hypotheses(model, sample)

        model.train(True) # model_state

        # apply any criterion-specific modifications to the sample/hypotheses
        sample, hypos = self.prepare_sample_and_hypotheses(model, sample, hypos)

        # create a new sample out of the hypotheses
        sample = self._update_sample_with_hypos(sample, hypos)

        net_output = model(**sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True).view(
            sample['nsentences'], sample['num_hypos_per_batch'], -1, net_output[0].size(-1)
        )

        loss, sample_size, logging_output = self.sequence_forward(lprobs, model, sample)


        #loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        #sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        # logging_output = {
        #     'loss': loss.data,
        #     #'nll_loss': nll_loss.data,
        #     'ntokens': sample['ntokens'],
        #     'nsentences': sample['target'].size(0),
        #     'sample_size': sample_size,
        # }
        return loss, sample_size, logging_output

    # def compute_loss(self, model, net_output, sample, reduce=True):
    #     lprobs = model.get_normalized_probs(net_output, log_probs=True)
    #     lprobs = lprobs.view(-1, lprobs.size(-1))
    #     target = model.get_targets(sample, net_output).view(-1, 1)
    #     loss, nll_loss = label_smoothed_nll_loss(
    #         lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
    #     )
    #     return loss, nll_loss


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

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
