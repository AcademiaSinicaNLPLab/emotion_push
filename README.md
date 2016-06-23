# EmotionPush


## Environment Setting
```bash
git submodule update --init corpus word2vec
cd word2vec
# Followd word2vec/README.md to setup word2vec

```

## Source Tree Architecture
**word2vec**: [word vector modules](https://github.com/AcademiaSinicaNLPLab/word2vec), import this modules to load pre-trained word vectors.
**corpus**: raw corpus data and preprocessing script, read corpus/README.md for modre information.
**cache**: cached data, e.g. word vector for word of certain corpus (loading word vectors for all words every time is not a good idea).
**model**: trained model files. Running emotion push server requires at least one trained model.
**src**: source code, read src/README.md for more information.

