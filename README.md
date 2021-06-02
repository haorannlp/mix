

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

For complete training code, please refer to `training_command/`

--------------------------------------------

# Generation

### Single model

```shell
MODEL=./checkpoints_wmt16ro2en_teahcer_forcing_ce_seed_1111/

python generate.py ./data-bin/wmt16_ro_en --path  $MODEL/checkpoint_best.pt \
       --batch-size 512 --beam 5 --remove-bpe --quiet
```

### Average model
```shell
# First averaging the models; make sure you've re-named the top-5 checkpoints
# as checkpoint1.pt,...,checkpoint5.pt
python scripts/average_checkpoints.py --inputs $MODEL \
       --num-epoch-checkpoints 5 --checkpoint-upper-bound 5 --output $MODEL/top_5.pt

python generate.py ./data-bin/wmt16_ro_en --path $MODEL/top_5.pt \
       --batch-size 512 --beam 5 --remove-bpe --quiet
```


------------------------------
# Citation







































