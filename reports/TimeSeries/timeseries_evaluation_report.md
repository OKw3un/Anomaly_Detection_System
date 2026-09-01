# Zaman Serisi Modelleri Performans Raporu

Toplam Veri SayÄ±sÄ±: 509632
GerÃ§ek Anomali SayÄ±sÄ±: 443
Normal Veri SayÄ±sÄ±: 509189

## XGBOOST Performansı
- **Toplam Bulduğu Anomali Sayısı:** 173
- **Doğru Tespit Ettiği (True Positive):** 138
- **Yanlış Alarm (False Positive):** 35
- **Kaçırdığı Anomali (False Negative):** 305
- **Geri Çağırma (Recall):** %31.15
- **Hassasiyet (Precision):** %79.77

```text
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    509189
           1       0.80      0.31      0.45       443

    accuracy                           1.00    509632
   macro avg       0.90      0.66      0.72    509632
weighted avg       1.00      1.00      1.00    509632

```

## RANDOM_FOREST Performansı
- **Toplam Bulduğu Anomali Sayısı:** 55
- **Doğru Tespit Ettiği (True Positive):** 53
- **Yanlış Alarm (False Positive):** 2
- **Kaçırdığı Anomali (False Negative):** 390
- **Geri Çağırma (Recall):** %11.96
- **Hassasiyet (Precision):** %96.36

```text
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    509189
           1       0.96      0.12      0.21       443

    accuracy                           1.00    509632
   macro avg       0.98      0.56      0.61    509632
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
