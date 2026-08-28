# Zaman Serisi Modelleri Performans Raporu

Toplam Veri SayÄ±sÄ±: 509632
GerÃ§ek Anomali SayÄ±sÄ±: 443
Normal Veri SayÄ±sÄ±: 509189

## XGBOOST Performansı
- **Toplam Bulduğu Anomali Sayısı:** 706
- **Doğru Tespit Ettiği (True Positive):** 269
- **Yanlış Alarm (False Positive):** 437
- **Kaçırdığı Anomali (False Negative):** 174
- **Geri Çağırma (Recall):** %60.72
- **Hassasiyet (Precision):** %38.10

```text
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    509189
           1       0.38      0.61      0.47       443

    accuracy                           1.00    509632
   macro avg       0.69      0.80      0.73    509632
weighted avg       1.00      1.00      1.00    509632

```

## LIGHTGBM Performansı
- **Toplam Bulduğu Anomali Sayısı:** 536
- **Doğru Tespit Ettiği (True Positive):** 237
- **Yanlış Alarm (False Positive):** 299
- **Kaçırdığı Anomali (False Negative):** 206
- **Geri Çağırma (Recall):** %53.50
- **Hassasiyet (Precision):** %44.22

```text
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    509189
           1       0.44      0.53      0.48       443

    accuracy                           1.00    509632
   macro avg       0.72      0.77      0.74    509632
weighted avg       1.00      1.00      1.00    509632

```

## RANDOM_FOREST Performansı
- **Toplam Bulduğu Anomali Sayısı:** 693
- **Doğru Tespit Ettiği (True Positive):** 212
- **Yanlış Alarm (False Positive):** 481
- **Kaçırdığı Anomali (False Negative):** 231
- **Geri Çağırma (Recall):** %47.86
- **Hassasiyet (Precision):** %30.59

```text
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    509189
           1       0.31      0.48      0.37       443

    accuracy                           1.00    509632
   macro avg       0.65      0.74      0.69    509632
weighted avg       1.00      1.00      1.00    509632

```

## ISOLATION_FOREST Performansı
- **Toplam Bulduğu Anomali Sayısı:** 34817
- **Doğru Tespit Ettiği (True Positive):** 161
- **Yanlış Alarm (False Positive):** 34656
- **Kaçırdığı Anomali (False Negative):** 282
- **Geri Çağırma (Recall):** %36.34
- **Hassasiyet (Precision):** %0.46

```text
              precision    recall  f1-score   support

           0       1.00      0.93      0.96    509189
           1       0.00      0.36      0.01       443

    accuracy                           0.93    509632
   macro avg       0.50      0.65      0.49    509632
weighted avg       1.00      0.93      0.96    509632

```

## ECOD Performansı
- **Toplam Bulduğu Anomali Sayısı:** 35374
- **Doğru Tespit Ettiği (True Positive):** 133
- **Yanlış Alarm (False Positive):** 35241
- **Kaçırdığı Anomali (False Negative):** 310
- **Geri Çağırma (Recall):** %30.02
- **Hassasiyet (Precision):** %0.38

```text
              precision    recall  f1-score   support

           0       1.00      0.93      0.96    509189
           1       0.00      0.30      0.01       443

    accuracy                           0.93    509632
   macro avg       0.50      0.62      0.49    509632
weighted avg       1.00      0.93      0.96    509632

```

## PCA Performansı
- **Toplam Bulduğu Anomali Sayısı:** 33612
- **Doğru Tespit Ettiği (True Positive):** 419
- **Yanlış Alarm (False Positive):** 33193
- **Kaçırdığı Anomali (False Negative):** 24
- **Geri Çağırma (Recall):** %94.58
- **Hassasiyet (Precision):** %1.25

```text
              precision    recall  f1-score   support

           0       1.00      0.93      0.97    509189
           1       0.01      0.95      0.02       443

    accuracy                           0.93    509632
   macro avg       0.51      0.94      0.50    509632
weighted avg       1.00      0.93      0.97    509632

```

## COPOD Performansı
- **Toplam Bulduğu Anomali Sayısı:** 33394
- **Doğru Tespit Ettiği (True Positive):** 149
- **Yanlış Alarm (False Positive):** 33245
- **Kaçırdığı Anomali (False Negative):** 294
- **Geri Çağırma (Recall):** %33.63
- **Hassasiyet (Precision):** %0.45

```text
              precision    recall  f1-score   support

           0       1.00      0.93      0.97    509189
           1       0.00      0.34      0.01       443

    accuracy                           0.93    509632
   macro avg       0.50      0.64      0.49    509632
weighted avg       1.00      0.93      0.97    509632

```

## HBOS Performansı
- **Toplam Bulduğu Anomali Sayısı:** 16108
- **Doğru Tespit Ettiği (True Positive):** 25
- **Yanlış Alarm (False Positive):** 16083
- **Kaçırdığı Anomali (False Negative):** 418
- **Geri Çağırma (Recall):** %5.64
- **Hassasiyet (Precision):** %0.16

```text
              precision    recall  f1-score   support

           0       1.00      0.97      0.98    509189
           1       0.00      0.06      0.00       443

    accuracy                           0.97    509632
   macro avg       0.50      0.51      0.49    509632
weighted avg       1.00      0.97      0.98    509632

```

## DEEP_SVDD Performansı
- **Toplam Bulduğu Anomali Sayısı:** 123091
- **Doğru Tespit Ettiği (True Positive):** 378
- **Yanlış Alarm (False Positive):** 122713
- **Kaçırdığı Anomali (False Negative):** 65
- **Geri Çağırma (Recall):** %85.33
- **Hassasiyet (Precision):** %0.31

```text
              precision    recall  f1-score   support

           0       1.00      0.76      0.86    509189
           1       0.00      0.85      0.01       443

    accuracy                           0.76    509632
   macro avg       0.50      0.81      0.43    509632
weighted avg       1.00      0.76      0.86    509632

```

## LSTM_AUTOENCODER Performansı
- **Toplam Bulduğu Anomali Sayısı:** 444
- **Doğru Tespit Ettiği (True Positive):** 0
- **Yanlış Alarm (False Positive):** 444
- **Kaçırdığı Anomali (False Negative):** 443
- **Geri Çağırma (Recall):** %0.00
- **Hassasiyet (Precision):** %0.00

```text
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    509189
           1       0.00      0.00      0.00       443

    accuracy                           1.00    509632
   macro avg       0.50      0.50      0.50    509632
weighted avg       1.00      1.00      1.00    509632

```

## SUPERVISED_XGBOOST Performansı
- **Toplam Bulduğu Anomali Sayısı:** 145
- **Doğru Tespit Ettiği (True Positive):** 94
- **Yanlış Alarm (False Positive):** 51
- **Kaçırdığı Anomali (False Negative):** 349
- **Geri Çağırma (Recall):** %21.22
- **Hassasiyet (Precision):** %64.83

```text
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    509189
           1       0.65      0.21      0.32       443

    accuracy                           1.00    509632
   macro avg       0.82      0.61      0.66    509632
weighted avg       1.00      1.00      1.00    509632

```

## SUPERVISED_LIGHTGBM Performansı
- **Toplam Bulduğu Anomali Sayısı:** 213
- **Doğru Tespit Ettiği (True Positive):** 136
- **Yanlış Alarm (False Positive):** 77
- **Kaçırdığı Anomali (False Negative):** 307
- **Geri Çağırma (Recall):** %30.70
- **Hassasiyet (Precision):** %63.85

```text
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    509189
           1       0.64      0.31      0.41       443

    accuracy                           1.00    509632
   macro avg       0.82      0.65      0.71    509632
weighted avg       1.00      1.00      1.00    509632

```
