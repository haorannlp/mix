#/bin/sh

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=examples/translation/mosesdecoder

# suffix of target language files
lng=en

sed 's/\@\@ //g' | \
$mosesdecoder/scripts/tokenizer/detokenizer.perl -l $lng
