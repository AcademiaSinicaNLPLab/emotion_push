# Process

## 
word2vec: word vector modules, import this modules to load pre-trained word vectors
corpus: raw corpus data and preprocessing script
cache: cached data, e.g. word vector for word of certain corpus (loading word vectors for all words every time is not a good idea)
model: trained model files. Running emotion push server requires at least one trained model
src: source code

## Setting up the environment
```bash
git submodule update --init corpus word2vec
cd word2vec
# Followd word2vec/README.md to setup word2vec
```
## Build Classifiers
- Write new classifier that is scikit-learn compatible in src/classifier.
- Train the classifier with src/*_train.py and store it in model/ (please refer to src/svm_train.py)

## Run Server
- For releasing, execute src/server.py
- For debugging, execute src/server.py -d

## Test Client& Server connection
- For releasing, execute src/client_test.py
- For debugging, execute src/client_test.py -d

# Server & Client Protocol
1. (GET) /litsmodel

```json
Input:
None
Output:
{"model1_name":["emotion_A", "emotion_B", "emotion_C"], "model2_name":["emotion_D", "emotion_E"]}
```

2. (POST) /predict
The output are the probability (or something similar) for each emotion belonging to the queried model name
```json
Input:
{"model":"model1_name", "text": "whatever text here"}
Output:
{"res": [0.2, 0.3, 0.9]} 
```

3. (POST) /log
```json
Input:
{"Whatever":"Whatever json object"}
Output:
None
```
