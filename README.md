

# Mixed Cross Entropy Loss for Neural Machine Translation

----------------------------------------

# Requirements and Installation
 
Our implementation is based on the implemetation of [OR-Transfomer](https://github.com/ictnlp/OR-NMT) and [Fairseq 0.9.0](https://github.com/pytorch/fairseq).

The code has been tested in the following enviroment:

- Ubuntu 18.04.4 LTS
- Python == 3.7

To install:

- `conda create -n mix python=3.7` 
- `conda activate mix`
- `git clone https://github.com/haorannlp/mix`
- `cd mix`
- `pip install -r requirements.txt`
- `pip install --editable .`

---------------------------------------

# Data Preparation

### WMT'16 Ro-En

- Downlaod `WMT'16 En-Ro` data from https://github.com/nyu-dl/dl4mt-nonauto
- Create a folder named `wmt16_ro_en` under `examples/translation/`
- Extract the `corpus.bpe.en/ro, dev.bpe.en/ro, test.bpe.en/ro` to the the folder created above
- ```shell
  TEXT=examples/translation/wmt16_ro_en
  # run the following command under "mix" directory
  fairseq-preprocess  --source-lang ro --target-lang en 
        --trainpref $TEXT/corpus.bpe --validpref  $TEXT/dev.bpe --testpref $TEXT/test.bpe 
        --destdir data-bin/wmt16_ro_en --thresholdtgt 0 --thresholdsrc 0 
        --workers 20
  ```

### WMT'16 Ru-En

- `cd examples/translation`
- Get the link to download `1mcorpus.zip` from https://translate.yandex.ru/corpus?lang=en
- `mkdir orig_wmt16ru2en`,   put `1mcorpus.zip` in this folder and `unzip 1mcorpus.zip`
- `bash prepare-wmt16ru2en.sh` (we did not include the `wiki-titles` dataset)
- ```shell
  TEXT=examples/translation/wmt16_ru_en
  # run the following command under "mix" directory
  fairseq-preprocess  --source-lang ru --target-lang en 
        --trainpref $TEXT/train --validpref  $TEXT/valid --testpref $TEXT/test 
        --destdir data-bin/wmt16_ru_en --thresholdtgt 0 --thresholdsrc 0 
        --workers 20
  ```

### WMT'14 En-De

- `cd examples/translation`
- `bash prepare-wmt14en2de-joint.sh --icml17` (we use `newstest2013` as dev set)
- ```shell
  TEXT=examples/translation/wmt14_en_de_joint
  # run the following command under "mix" directory
  fairseq-preprocess  --source-lang en --target-lang de 
        --trainpref $TEXT/train --validpref  $TEXT/valid --testpref $TEXT/test 
        --destdir data-bin/wmt14_en_de --thresholdtgt 0 --thresholdsrc 0 
        --workers 20
  ```

-------------------------------------

# Training 
We use random seeds `1111,2222,3333` for WMT'16 Ro-En, WMT'16 Ru-En, random seeds `1,2,3` for WMT'14 
En-De.

### WMT'16 Ro-En

#### Teacher Forcing + CE
```shell
python train.py data-bin/wmt16_ro_en --warmup-init-lr 1e-07 --warmup-updates 1000 
  --lr 0.0007  --lr-scheduler reduce_lr_on_plateau  --lr-shrink 0.5 --mode max 
  --lr_schedule_patience 4 --min-lr 1e-09  --optimizer adam --adam-betas '(0.9, 0.98)'  
  --criterion oracle_label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 
  --clip-norm 0.0  --max-tokens 12288 --max-update 8000 --arch oracle_transformer_wmt_en_de 
  --eval-bleu --eval-bleu-args '{"beam":3,"max_len_a":1.2,"max_len_b":10}' 
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
  --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples 
  --use-old-adam --seed 1111 --distributed-port 22 --distributed-world-size 2 
  --ddp-backend=no_c10d --keep-last-epochs 15   --fp16 --update-freq 2 
  --save-dir ./checkpoints_wmt16ro2en_teahcer_forcing_ce_seed_1111/ 
```

#### Teacher Forcing + mixed CE
```shell
python train.py data-bin/wmt16_ro_en --warmup-init-lr 1e-07 --warmup-updates 1000 
    --lr 0.0007  --lr-scheduler reduce_lr_on_plateau  --lr-shrink 0.5 --mode max  
    --lr_schedule_patience 4 --min-lr 1e-09  --optimizer adam --adam-betas '(0.9, 0.98)'   
    --criterion oracle_label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001  
    --clip-norm 0.0  --max-tokens 12288 --max-update 8000 --arch oracle_transformer_wmt_en_de 
    --eval-bleu --eval-bleu-args '{"beam":3,"max_len_a":1.2,"max_len_b":10}' 
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
    --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples  
    --use-old-adam --seed 1111 --distributed-port 22 --distributed-world-size 2  
    --ddp-backend=no_c10d --keep-last-epochs 15 --fp16 --update-freq 2 
    --use-mix-CE --greedy-mix-CE --decay-k 8000 
    --save-dir ./checkpoints_wmt16ro2en_teahcer_forcing_mixed_ce_seed_1111/ 
```

#### Scheduled Sampling + CE
```shell
python train.py data-bin/wmt16_ro_en --warmup-init-lr 1e-07 --warmup-updates 1000 
  --lr 0.0007  --lr-scheduler reduce_lr_on_plateau  --lr-shrink 0.5 --mode max  
  --lr_schedule_patience 4 --min-lr 1e-09  --optimizer adam --adam-betas '(0.9, 0.98)' 
  --criterion oracle_label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 
  --clip-norm 0.0  --max-tokens 12288 --max-update 8000 --arch oracle_transformer_wmt_en_de 
  --eval-bleu --eval-bleu-args '{"beam":3,"max_len_a":1.2,"max_len_b":10}' 
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
  --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples 
  --use-old-adam --seed 1111 --distributed-port 22 --distributed-world-size 2 
  --ddp-backend=no_c10d --keep-last-epochs 15 --fp16 --update-freq 2
  --use-word-level-oracles  --ss-exponential 0.7 --decay-k 8000 
   --save-dir ./checkpoints_wmt16ro2en_scheduled_sampling_ce_seed_1111/
```

#### Scheduled Sampling + mixed CE
```shell
python train.py data-bin/wmt16_ro_en --warmup-init-lr 1e-07 --warmup-updates 1000 
  --lr 0.0007  --lr-scheduler reduce_lr_on_plateau  --lr-shrink 0.5 --mode max  
  --lr_schedule_patience 4 --min-lr 1e-09  --optimizer adam --adam-betas '(0.9, 0.98)'
  --criterion oracle_label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001
  --clip-norm 0.0  --max-tokens 12288 --max-update 8000 --arch oracle_transformer_wmt_en_de
  --eval-bleu --eval-bleu-args '{"beam":3,"max_len_a":1.2,"max_len_b":10}'
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
  --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples
  --use-old-adam --seed 1111 --distributed-port 22 --distributed-world-size 2
  --ddp-backend=no_c10d --keep-last-epochs 15 --fp16 --update-freq 2 
  --use-mix-CE --word-oracle-noise-greedy-output --ss-exponential 0.7 --decay-k 8000 
  --save-dir ./checkpoints_wmt16ro2en_scheduled_sampling_mixed_ce_seed_1111/ 
```

#### Word Oracle + CE
```shell
python train.py data-bin/wmt16_ro_en --warmup-init-lr 1e-07 --warmup-updates 1000 
  --lr 0.0007  --lr-scheduler reduce_lr_on_plateau  --lr-shrink 0.5 --mode max  
  --lr_schedule_patience 4 --min-lr 1e-09  --optimizer adam --adam-betas '(0.9, 0.98)' 
  --criterion oracle_label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 
  --clip-norm 0.0  --max-tokens 12288 --max-update 8000 --arch oracle_transformer_wmt_en_de 
  --eval-bleu --eval-bleu-args '{"beam":3,"max_len_a":1.2,"max_len_b":10}' 
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
  --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples 
  --use-old-adam --seed 1111 --distributed-port 22 --distributed-world-size 2 
  --ddp-backend=no_c10d --keep-last-epochs 15 --fp16 --update-freq 2
  --use-word-level-oracles --use-greedy-gumbel-noise --gumbel-noise 0.5
  --ss-exponential 0.7 --decay-k 8000 
  --save-dir ./checkpoints_wmt16ro2en_word_oracle_ce_seed_1111/
```

#### Word Oracle + mixed CE
```shell
python train.py data-bin/wmt16_ro_en --warmup-init-lr 1e-07 --warmup-updates 1000 
  --lr 0.0007  --lr-scheduler reduce_lr_on_plateau  --lr-shrink 0.5 --mode max  
  --lr_schedule_patience 4 --min-lr 1e-09  --optimizer adam --adam-betas '(0.9, 0.98)'
  --criterion oracle_label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001
  --clip-norm 0.0  --max-tokens 12288 --max-update 8000 --arch oracle_transformer_wmt_en_de
  --eval-bleu --eval-bleu-args '{"beam":3,"max_len_a":1.2,"max_len_b":10}'
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
  --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples
  --use-old-adam --seed 1111 --distributed-port 22 --distributed-world-size 2
  --ddp-backend=no_c10d --keep-last-epochs 15 --fp16 --update-freq 2 
  --use-mix-CE --word-oracle-noise-greedy-output --use-greedy-gumbel-noise --gumbel-noise 0.5
  --ss-exponential 0.7 --decay-k 8000 
  --save-dir ./checkpoints_wmt16ro2en_word_oracle_mixed_ce_seed_1111/ 
```


### WMT'16 Ru-En

#### Teacher Forcing + CE
```shell
python train.py data-bin/wmt16_ru_en --warmup-init-lr 1e-07 --warmup-updates 4000 
  --lr 0.0007  --lr-scheduler reduce_lr_on_plateau  --lr-shrink 0.5 --mode max  
  --lr_schedule_patience 4 --min-lr 1e-09  --optimizer adam --adam-betas '(0.9, 0.98)'
  --criterion oracle_label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001
  --clip-norm 0.0  --max-tokens 12288 --max-update 45000 --arch oracle_transformer_wmt_en_de
  --eval-bleu --eval-bleu-args '{"beam":3,"max_len_a":1.2,"max_len_b":10}'
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
  --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples 
  --use-old-adam --seed 1111 --distributed-port 22 --distributed-world-size 2 
  --ddp-backend=no_c10d --keep-last-epochs 20  --fp16 --update-freq 2 
  --save-dir ./checkpoints_wmt16ru2en_teacher_forcing_ce_seed_1111/
```

#### Teacher Forcing + mixed CE
```shell
python train.py data-bin/wmt16_ru_en --warmup-init-lr 1e-07 --warmup-updates 4000 
  --lr 0.0007  --lr-scheduler reduce_lr_on_plateau  --lr-shrink 0.5 --mode max  
  --lr_schedule_patience 4 --min-lr 1e-09  --optimizer adam --adam-betas '(0.9, 0.98)'
  --criterion oracle_label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001
  --clip-norm 0.0  --max-tokens 12288 --max-update 45000 --arch oracle_transformer_wmt_en_de
  --eval-bleu --eval-bleu-args '{"beam":3,"max_len_a":1.2,"max_len_b":10}'
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
  --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples 
  --use-old-adam --seed 1111 --distributed-port 22 --distributed-world-size 2 
  --ddp-backend=no_c10d --keep-last-epochs 20  --fp16 --update-freq 2 
  --use-mix-CE --greedy-mix-CE --decay-k 45000
  --save-dir ./checkpoints_wmt16ru2en_teacher_forcing_mixed_ce_seed_1111/
```

#### Scheduled Sampling + CE
```shell
python train.py data-bin/wmt16_ru_en --warmup-init-lr 1e-07 --warmup-updates 4000 
  --lr 0.0007  --lr-scheduler reduce_lr_on_plateau  --lr-shrink 0.5 --mode max  
  --lr_schedule_patience 4 --min-lr 1e-09  --optimizer adam --adam-betas '(0.9, 0.98)' 
  --criterion oracle_label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 
  --clip-norm 0.0  --max-tokens 12288 --max-update 45000 --arch oracle_transformer_wmt_en_de 
  --eval-bleu --eval-bleu-args '{"beam":3,"max_len_a":1.2,"max_len_b":10}' 
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
  --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples 
  --use-old-adam --seed 1111 --distributed-port 22 --distributed-world-size 2 
  --ddp-backend=no_c10d --keep-last-epochs 20 --fp16 --update-freq 2
  --use-word-level-oracles  --ss-exponential 0.8 --decay-k 45000 
  --save-dir ./checkpoints_wmt16ru2en_scheduled_sampling_ce_seed_1111/
```

#### Scheduled Sampling + mixed CE
```shell
python train.py data-bin/wmt16_ru_en --warmup-init-lr 1e-07 --warmup-updates 4000 
  --lr 0.0007  --lr-scheduler reduce_lr_on_plateau  --lr-shrink 0.5 --mode max  
  --lr_schedule_patience 4 --min-lr 1e-09  --optimizer adam --adam-betas '(0.9, 0.98)'
  --criterion oracle_label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001
  --clip-norm 0.0  --max-tokens 12288 --max-update 45000 --arch oracle_transformer_wmt_en_de
  --eval-bleu --eval-bleu-args '{"beam":3,"max_len_a":1.2,"max_len_b":10}'
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
  --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples
  --use-old-adam --seed 1111 --distributed-port 22 --distributed-world-size 2
  --ddp-backend=no_c10d --keep-last-epochs 20 --fp16 --update-freq 2
  --use-mix-CE --word-oracle-noise-greedy-output --ss-exponential 0.8 --decay-k 45000 
  --save-dir ./checkpoints_wmt16ru2en_scheduled_sampling_mixed_ce_seed_1111/ 
```

#### Word Oracle + CE
```shell
python train.py data-bin/wmt16_ru_en --warmup-init-lr 1e-07 --warmup-updates 4000 
  --lr 0.0007  --lr-scheduler reduce_lr_on_plateau  --lr-shrink 0.5 --mode max  
  --lr_schedule_patience 4 --min-lr 1e-09  --optimizer adam --adam-betas '(0.9, 0.98)' 
  --criterion oracle_label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 
  --clip-norm 0.0  --max-tokens 12288 --max-update 45000 --arch oracle_transformer_wmt_en_de 
  --eval-bleu --eval-bleu-args '{"beam":3,"max_len_a":1.2,"max_len_b":10}' 
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
  --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples 
  --use-old-adam --seed 1111 --distributed-port 22 --distributed-world-size 2 
  --ddp-backend=no_c10d --keep-last-epochs 20 --fp16 --update-freq 2
  --use-word-level-oracles --use-greedy-gumbel-noise --gumbel-noise 0.5
  --ss-exponential 0.8 --decay-k 45000 
  --save-dir ./checkpoints_wmt16ru2en_word_oracle_ce_seed_1111/
```

#### Word Oracle + mixed CE
```shell
python train.py data-bin/wmt16_ru_en --warmup-init-lr 1e-07 --warmup-updates 4000 
  --lr 0.0007  --lr-scheduler reduce_lr_on_plateau  --lr-shrink 0.5 --mode max  
  --lr_schedule_patience 4 --min-lr 1e-09  --optimizer adam --adam-betas '(0.9, 0.98)'
  --criterion oracle_label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001
  --clip-norm 0.0  --max-tokens 12288 --max-update 45000 --arch oracle_transformer_wmt_en_de
  --eval-bleu --eval-bleu-args '{"beam":3,"max_len_a":1.2,"max_len_b":10}'
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
  --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples
  --use-old-adam --seed 1111 --distributed-port 22 --distributed-world-size 2
  --ddp-backend=no_c10d --keep-last-epochs 20 --fp16 --update-freq 2
  --use-mix-CE --word-oracle-noise-greedy-output --use-greedy-gumbel-noise --gumbel-noise 0.5
  --ss-exponential 0.8 --decay-k 45000 
  --save-dir ./checkpoints_wmt16ru2en_word_oracle_mixed_ce_seed_1111/ 
```

### WMT'14 En-De

#### Teacher Forcing + CE
```shell
python train.py data-bin/wmt14_en_de --warmup-init-lr 1e-07 --warmup-updates 4000 
  --lr 0.0007  --lr-scheduler reduce_lr_on_plateau  --lr-shrink 0.5 --mode max  
  --lr_schedule_patience 4 --min-lr 1e-09  --optimizer adam --adam-betas '(0.9, 0.98)'
  --criterion oracle_label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001
  --clip-norm 0.0  --max-tokens 12288 --max-update 80000 --arch oracle_transformer_wmt_en_de
  --eval-bleu --eval-bleu-args '{"beam":3,"max_len_a":1.2,"max_len_b":10}'
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
  --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples 
  --use-old-adam --seed 1 --distributed-port 22 --distributed-world-size 4
  --ddp-backend=no_c10d --keep-last-epochs 40  --fp16 --update-freq 2 
  --save-dir ./checkpoints_wmt14en2de_teacher_forcing_ce_seed_1/
```

#### Teacher Forcing + mixed CE
```shell
python train.py data-bin/wmt14_en_de --warmup-init-lr 1e-07 --warmup-updates 4000 
  --lr 0.0007  --lr-scheduler reduce_lr_on_plateau  --lr-shrink 0.5 --mode max  
  --lr_schedule_patience 4 --min-lr 1e-09  --optimizer adam --adam-betas '(0.9, 0.98)'
  --criterion oracle_label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001
  --clip-norm 0.0  --max-tokens 12288 --max-update 80000 --arch oracle_transformer_wmt_en_de
  --eval-bleu --eval-bleu-args '{"beam":3,"max_len_a":1.2,"max_len_b":10}'
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
  --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples 
  --use-old-adam --seed 1 --distributed-port 22 --distributed-world-size 4
  --ddp-backend=no_c10d --keep-last-epochs 40  --fp16 --update-freq 2 
  --use-mix-CE --greedy-mix-CE --decay-k 80000
  --save-dir ./checkpoints_wmt14en2de_teacher_forcing_mixed_ce_seed_1/
```

#### Scheduled Sampling + CE
```shell
python train.py data-bin/wmt14_en_de --warmup-init-lr 1e-07 --warmup-updates 4000 
  --lr 0.0007  --lr-scheduler reduce_lr_on_plateau  --lr-shrink 0.5 --mode max  
  --lr_schedule_patience 4 --min-lr 1e-09  --optimizer adam --adam-betas '(0.9, 0.98)'
  --criterion oracle_label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001
  --clip-norm 0.0  --max-tokens 12288 --max-update 80000 --arch oracle_transformer_wmt_en_de
  --eval-bleu --eval-bleu-args '{"beam":3,"max_len_a":1.2,"max_len_b":10}'
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
  --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples 
  --use-old-adam --seed 1 --distributed-port 22 --distributed-world-size 4
  --ddp-backend=no_c10d --keep-last-epochs 40  --fp16 --update-freq 2 
  --use-word-level-oracles  --ss-exponential 0.8 --decay-k 80000 
  --save-dir ./checkpoints_wmt14en2de_scheduled_sampling_ce_seed_1/
```

#### Scheduled Sampling + mixed CE
```shell
python train.py data-bin/wmt14_en_de --warmup-init-lr 1e-07 --warmup-updates 4000 
  --lr 0.0007  --lr-scheduler reduce_lr_on_plateau  --lr-shrink 0.5 --mode max  
  --lr_schedule_patience 4 --min-lr 1e-09  --optimizer adam --adam-betas '(0.9, 0.98)'
  --criterion oracle_label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001
  --clip-norm 0.0  --max-tokens 12288 --max-update 80000 --arch oracle_transformer_wmt_en_de
  --eval-bleu --eval-bleu-args '{"beam":3,"max_len_a":1.2,"max_len_b":10}'
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
  --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples 
  --use-old-adam --seed 1 --distributed-port 22 --distributed-world-size 4
  --ddp-backend=no_c10d --keep-last-epochs 40  --fp16 --update-freq 2 
  --use-mix-CE --word-oracle-noise-greedy-output --ss-exponential 0.8 --decay-k 80000 
  --save-dir ./checkpoints_wmt14en2de_scheduled_sampling_mixed_ce_seed_1/ 
```

#### Word Oracle + CE
```shell
python train.py data-bin/wmt14_en_de --warmup-init-lr 1e-07 --warmup-updates 4000 
  --lr 0.0007  --lr-scheduler reduce_lr_on_plateau  --lr-shrink 0.5 --mode max  
  --lr_schedule_patience 4 --min-lr 1e-09  --optimizer adam --adam-betas '(0.9, 0.98)'
  --criterion oracle_label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001
  --clip-norm 0.0  --max-tokens 12288 --max-update 80000 --arch oracle_transformer_wmt_en_de
  --eval-bleu --eval-bleu-args '{"beam":3,"max_len_a":1.2,"max_len_b":10}'
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
  --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples 
  --use-old-adam --seed 1 --distributed-port 22 --distributed-world-size 4
  --ddp-backend=no_c10d --keep-last-epochs 40  --fp16 --update-freq 2 
  --use-word-level-oracles --use-greedy-gumbel-noise --gumbel-noise 0.5
  --ss-exponential 0.8 --decay-k 80000 
  --save-dir ./checkpoints_wmt14en2de_word_oracle_ce_seed_1/
```

#### Word Oracle + mixed CE
```shell
python train.py data-bin/wmt14_en_de --warmup-init-lr 1e-07 --warmup-updates 4000 
  --lr 0.0007  --lr-scheduler reduce_lr_on_plateau  --lr-shrink 0.5 --mode max  
  --lr_schedule_patience 4 --min-lr 1e-09  --optimizer adam --adam-betas '(0.9, 0.98)'
  --criterion oracle_label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001
  --clip-norm 0.0  --max-tokens 12288 --max-update 80000 --arch oracle_transformer_wmt_en_de
  --eval-bleu --eval-bleu-args '{"beam":3,"max_len_a":1.2,"max_len_b":10}'
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
  --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples 
  --use-old-adam --seed 1 --distributed-port 22 --distributed-world-size 4
  --ddp-backend=no_c10d --keep-last-epochs 40  --fp16 --update-freq 2 
  --use-mix-CE --word-oracle-noise-greedy-output --use-greedy-gumbel-noise --gumbel-noise 0.5
  --ss-exponential 0.8 --decay-k 80000 
  --save-dir ./checkpoints_wmt14en2de_word_oracle_mixed_ce_seed_1/ 
```




















