```
python split.py --task 1 --pred_path final_preds/test/subtask_1/pred_fas.csv --gold_path gold/fas.csv --out_detected fas_detected_t1.csv --out_missed fas_missed_t1.csv

python split.py --task 2 --pred_path final_preds/test/subtask_2/pred_fas.csv --gold_path gold/fas.csv --out_detected fas_detected_t2.csv --out_missed fas_missed_t2.csv

python split.py --task 3 --pred_path final_preds/test/subtask_3/pred_fas.csv --gold_path gold/fas.csv --out_detected fas_detected_t3.csv --out_missed fas_missed_t3.csv
```

# Persian

## Subtask 1

```
=== polarization ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[283 102]
 [102 997]]
TN=283 FP=102 FN=102 TP=997
Classification report:
              precision    recall  f1-score   support

           0     0.7351    0.7351    0.7351       385
           1     0.9072    0.9072    0.9072      1099

    accuracy                         0.8625      1484
   macro avg     0.8211    0.8211    0.8211      1484
weighted avg     0.8625    0.8625    0.8625      1484
```

## Subtask 2

```
=== political ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[650 183]
 [ 64 587]]
TN=650 FP=183 FN=64 TP=587
Classification report:
              precision    recall  f1-score   support

           0     0.9104    0.7803    0.8403       833
           1     0.7623    0.9017    0.8262       651

    accuracy                         0.8336      1484
   macro avg     0.8364    0.8410    0.8333      1484
weighted avg     0.8454    0.8336    0.8341      1484


=== racial/ethnic ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[1424   24]
 [  23   13]]
TN=1424 FP=24 FN=23 TP=13
Classification report:
              precision    recall  f1-score   support

           0     0.9841    0.9834    0.9838      1448
           1     0.3514    0.3611    0.3562        36

    accuracy                         0.9683      1484
   macro avg     0.6677    0.6723    0.6700      1484
weighted avg     0.9688    0.9683    0.9685      1484


=== religious ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[1303   38]
 [  38  105]]
TN=1303 FP=38 FN=38 TP=105
Classification report:
              precision    recall  f1-score   support

           0     0.9717    0.9717    0.9717      1341
           1     0.7343    0.7343    0.7343       143

    accuracy                         0.9488      1484
   macro avg     0.8530    0.8530    0.8530      1484
weighted avg     0.9488    0.9488    0.9488      1484


=== gender/sexual ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[1360   35]
 [  35   54]]
TN=1360 FP=35 FN=35 TP=54
Classification report:
              precision    recall  f1-score   support

           0     0.9749    0.9749    0.9749      1395
           1     0.6067    0.6067    0.6067        89

    accuracy                         0.9528      1484
   macro avg     0.7908    0.7908    0.7908      1484
weighted avg     0.9528    0.9528    0.9528      1484


=== other ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[1021  104]
 [ 112  247]]
TN=1021 FP=104 FN=112 TP=247
Classification report:
              precision    recall  f1-score   support

           0     0.9011    0.9076    0.9043      1125
           1     0.7037    0.6880    0.6958       359

    accuracy                         0.8544      1484
   macro avg     0.8024    0.7978    0.8001      1484
weighted avg     0.8534    0.8544    0.8539      1484
```

## Subtask 3

```
=== stereotype ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[1068  221]
 [  87  108]]
TN=1068 FP=221 FN=87 TP=108
Classification report:
              precision    recall  f1-score   support

           0     0.9247    0.8285    0.8740      1289
           1     0.3283    0.5538    0.4122       195

    accuracy                         0.7925      1484
   macro avg     0.6265    0.6912    0.6431      1484
weighted avg     0.8463    0.7925    0.8133      1484


=== vilification ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[264 353]
 [ 53 814]]
TN=264 FP=353 FN=53 TP=814
Classification report:
              precision    recall  f1-score   support

           0     0.8328    0.4279    0.5653       617
           1     0.6975    0.9389    0.8004       867

    accuracy                         0.7264      1484
   macro avg     0.7652    0.6834    0.6829      1484
weighted avg     0.7538    0.7264    0.7027      1484


=== dehumanization ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[1395   25]
 [  41   23]]
TN=1395 FP=25 FN=41 TP=23
Classification report:
              precision    recall  f1-score   support

           0     0.9714    0.9824    0.9769      1420
           1     0.4792    0.3594    0.4107        64

    accuracy                         0.9555      1484
   macro avg     0.7253    0.6709    0.6938      1484
weighted avg     0.9502    0.9555    0.9525      1484


=== extreme_language ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[1069  165]
 [ 128  122]]
TN=1069 FP=165 FN=128 TP=122
Classification report:
              precision    recall  f1-score   support

           0     0.8931    0.8663    0.8795      1234
           1     0.4251    0.4880    0.4544       250

    accuracy                         0.8026      1484
   macro avg     0.6591    0.6771    0.6669      1484
weighted avg     0.8142    0.8026    0.8079      1484


=== lack_of_empathy ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[1295   43]
 [  85   61]]
TN=1295 FP=43 FN=85 TP=61
Classification report:
              precision    recall  f1-score   support

           0     0.9384    0.9679    0.9529      1338
           1     0.5865    0.4178    0.4880       146

    accuracy                         0.9137      1484
   macro avg     0.7625    0.6928    0.7205      1484
weighted avg     0.9038    0.9137    0.9072      1484


=== invalidation ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[1247  119]
 [  60   58]]
TN=1247 FP=119 FN=60 TP=58
Classification report:
              precision    recall  f1-score   support

           0     0.9541    0.9129    0.9330      1366
           1     0.3277    0.4915    0.3932       118

    accuracy                         0.8794      1484
   macro avg     0.6409    0.7022    0.6631      1484
weighted avg     0.9043    0.8794    0.8901      1484
```



# English

## Subtask 1

```
=== polarization ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[788 131]
 [142 391]]
TN=788 FP=131 FN=142 TP=391
Classification report:
              precision    recall  f1-score   support

           0     0.8473    0.8575    0.8524       919
           1     0.7490    0.7336    0.7412       533

    accuracy                         0.8120      1452
   macro avg     0.7982    0.7955    0.7968      1452
weighted avg     0.8112    0.8120    0.8116      1452
```

## Subtask 2

```
=== political ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[721 213]
 [ 74 444]]
TN=721 FP=213 FN=74 TP=444
Classification report:
              precision    recall  f1-score   support

           0     0.9069    0.7719    0.8340       934
           1     0.6758    0.8571    0.7557       518

    accuracy                         0.8023      1452
   macro avg     0.7914    0.8145    0.7949      1452
weighted avg     0.8245    0.8023    0.8061      1452


=== racial/ethnic ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[1250   75]
 [  56   71]]
TN=1250 FP=75 FN=56 TP=71
Classification report:
              precision    recall  f1-score   support

           0     0.9571    0.9434    0.9502      1325
           1     0.4863    0.5591    0.5201       127

    accuracy                         0.9098      1452
   macro avg     0.7217    0.7512    0.7352      1452
weighted avg     0.9159    0.9098    0.9126      1452


=== religious ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[1370   31]
 [  26   25]]
TN=1370 FP=31 FN=26 TP=25
Classification report:
              precision    recall  f1-score   support

           0     0.9814    0.9779    0.9796      1401
           1     0.4464    0.4902    0.4673        51

    accuracy                         0.9607      1452
   macro avg     0.7139    0.7340    0.7235      1452
weighted avg     0.9626    0.9607    0.9616      1452


=== gender/sexual ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[1389   30]
 [  18   15]]
TN=1389 FP=30 FN=18 TP=15
Classification report:
              precision    recall  f1-score   support

           0     0.9872    0.9789    0.9830      1419
           1     0.3333    0.4545    0.3846        33

    accuracy                         0.9669      1452
   macro avg     0.6603    0.7167    0.6838      1452
weighted avg     0.9723    0.9669    0.9694      1452


=== other ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[1359   35]
 [  41   17]]
TN=1359 FP=35 FN=41 TP=17
Classification report:
              precision    recall  f1-score   support

           0     0.9707    0.9749    0.9728      1394
           1     0.3269    0.2931    0.3091        58

    accuracy                         0.9477      1452
   macro avg     0.6488    0.6340    0.6409      1452
weighted avg     0.9450    0.9477    0.9463      1452
```



## Subtask 3

```
=== stereotype ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[1094  139]
 [ 107  112]]
TN=1094 FP=139 FN=107 TP=112
Classification report:
              precision    recall  f1-score   support

           0     0.9109    0.8873    0.8989      1233
           1     0.4462    0.5114    0.4766       219

    accuracy                         0.8306      1452
   macro avg     0.6786    0.6993    0.6878      1452
weighted avg     0.8408    0.8306    0.8352      1452


=== vilification ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[888 189]
 [110 265]]
TN=888 FP=189 FN=110 TP=265
Classification report:
              precision    recall  f1-score   support

           0     0.8898    0.8245    0.8559      1077
           1     0.5837    0.7067    0.6393       375

    accuracy                         0.7941      1452
   macro avg     0.7367    0.7656    0.7476      1452
weighted avg     0.8107    0.7941    0.8000      1452


=== dehumanization ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[1128  148]
 [  86   90]]
TN=1128 FP=148 FN=86 TP=90
Classification report:
              precision    recall  f1-score   support

           0     0.9292    0.8840    0.9060      1276
           1     0.3782    0.5114    0.4348       176

    accuracy                         0.8388      1452
   macro avg     0.6537    0.6977    0.6704      1452
weighted avg     0.8624    0.8388    0.8489      1452


=== extreme_language ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[876 231]
 [ 81 264]]
TN=876 FP=231 FN=81 TP=264
Classification report:
              precision    recall  f1-score   support

           0     0.9154    0.7913    0.8488      1107
           1     0.5333    0.7652    0.6286       345

    accuracy                         0.7851      1452
   macro avg     0.7243    0.7783    0.7387      1452
weighted avg     0.8246    0.7851    0.7965      1452


=== lack_of_empathy ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[1122  169]
 [  85   76]]
TN=1122 FP=169 FN=85 TP=76
Classification report:
              precision    recall  f1-score   support

           0     0.9296    0.8691    0.8983      1291
           1     0.3102    0.4720    0.3744       161

    accuracy                         0.8251      1452
   macro avg     0.6199    0.6706    0.6364      1452
weighted avg     0.8609    0.8251    0.8402      1452


=== invalidation ===
Confusion matrix (rows=gold [0,1], cols=pred [0,1])
[[926 262]
 [ 94 170]]
TN=926 FP=262 FN=94 TP=170
Classification report:
              precision    recall  f1-score   support

           0     0.9078    0.7795    0.8388      1188
           1     0.3935    0.6439    0.4885       264

    accuracy                         0.7548      1452
   macro avg     0.6507    0.7117    0.6636      1452
weighted avg     0.8143    0.7548    0.7751      1452
```
