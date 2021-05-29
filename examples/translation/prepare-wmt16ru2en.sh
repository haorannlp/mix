#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_EN_TOKENS=24000
BPE_RU_TOKENS=24000

URLS=(
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://data.statmt.org/wmt16/translation-task/training-parallel-nc-v11.tgz"
    #"http://www.statmt.org/wmt15/wiki-titles.tgz"
    "https://translate.yandex.ru/corpus?lang=en"
    "http://data.statmt.org/wmt16/translation-task/dev.tgz"
    "http://data.statmt.org/wmt16/translation-task/test.tgz"
)
FILES=(
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v11.tgz"
    #"wiki-titles.tgz"
    "1mcorpus.zip"
    "dev.tgz"
    "test.tgz"
)
CORPORA=(
    "commoncrawl.ru-en"
    "training-parallel-nc-v11/news-commentary-v11.ru-en"
    "corpus.en_ru.1m"
)

# This will make the dataset compatible to the one used in "Convolutional Sequence to Sequence Learning"
# https://arxiv.org/abs/1705.03122
#if [ "$1" == "--icml17" ]; then
#    URLS[2]="http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
#    FILES[2]="training-parallel-nc-v9.tgz"
#    CORPORA[2]="training/news-commentary-v9.de-en"
#    OUTDIR=wmt14_en_de
#else
#    OUTDIR=wmt17_en_de
#fi

OUTDIR=wmt16_ru_en

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=ru
tgt=en
lang=ru-en
prep=$OUTDIR
tmp=$prep/tmp
orig=orig_wmt16ru2en
dev=dev/newstest2013

mkdir -p $orig $tmp $prep

cd $orig

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"

    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
	elif [ ${file: -4} == ".zip"]; then
	    unzip $file
        fi
    fi
done
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
    done
done

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/test/newstest2016-ruen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
    echo ""
done


echo "pre-processing valid data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/dev/newstest2015-ruen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -a -l $l > $tmp/valid.$l
    echo ""
done


echo "splitting train and valid..."
for l in $src $tgt; do
    #awk '{if (NR%100 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
    awk '{if (NR%100 != -100000000)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
done


TRAIN=$tmp/train.ru-en
BPE_EN_CODE=$prep/en_code
BPE_RU_CODE=$prep/ru_code
rm -f $TRAIN
#for l in $src $tgt; do
#    cat $tmp/train.$l >> $TRAIN
#done


echo "learn_bpe.py on train.en..."
python $BPEROOT/learn_bpe.py -s $BPE_EN_TOKENS < $tmp/train.en > $BPE_EN_CODE
echo "learn_bpe.py on train.ru..."
python $BPEROOT/learn_bpe.py -s $BPE_RU_TOKENS < $tmp/train.ru > $BPE_RU_CODE

for f in train.en valid.en test.en; do
    echo "apply_bpe.py to ${f}..."
    python $BPEROOT/apply_bpe.py -c $BPE_EN_CODE < $tmp/$f > $tmp/bpe.$f
done

for f in train.ru valid.ru test.ru; do
    echo "apply_bpe.py to ${f}..."
    python $BPEROOT/apply_bpe.py -c $BPE_RU_CODE < $tmp/$f > $tmp/bpe.$f
done


perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
#perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250
for L in $src $tgt; do
    cp $tmp/bpe.valid.$L $prep/valid.$L
done

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done
