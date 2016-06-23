## Run Server
```bash
./server.py
```

## Build Classifiers
- Write new classifier that is scikit-learn compatible in src/classifier.
- Train the classifier with src/*_train.py and store it in model/ (please refer to src/svm_train.py)

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
