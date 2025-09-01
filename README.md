# 🌳 Decision Trees (Karar Ağaçları) - Türkçe Rehber

Makine öğrenmesinin en sezgisel algoritması olan **Karar Ağaçları** hakkında kapsamlı Türkçe rehber.

## 📖 Proje Hakkında

Bu proje, **"Hands-On Machine Learning with Scikit-Learn and TensorFlow"** kitabının 6. bölümünden esinlenerek hazırlanmış, Türkçe bir öğretici rehberdir. Hem teorik bilgileri hem de pratik uygulamaları içerir.

## 🎯 İçerik

- **Temel Kavramlar**: Karar ağaçları nasıl çalışır?
- **CART Algoritması**: Scikit-learn'ün kullandığı algoritmanın iç yapısı
- **Gini Impurity**: Safsızlık ölçümü ve hesaplama yöntemleri
- **Pratik Örnekler**: İris veri seti ve kredi onayı senaryosu
- **Hiperparametre Optimizasyonu**: GridSearchCV ile en iyi parametreleri bulma
- **Model Değerlendirme**: Confusion matrix, classification report
- **Görselleştirme**: Ağaç yapılarını ve sonuçları görsel olarak anlama

## 🚀 Kullanım

### Gereksinimler
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
```

### Hızlı Başlangıç
```python
# Veri yükle
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# Model eğit
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# Tahmin yap
predictions = model.predict(X)
```

## 📊 Örnekler

Projede şu örnekler bulunur:

1. **İris Sınıflandırması**: Temel karar ağacı uygulaması
2. **Manuel Gini Hesaplama**: Algoritmanın iç yapısını anlama
3. **Hiperparametre Tuning**: En iyi parametreleri bulma
4. **Kredi Onayı Sistemi**: Gerçek dünya uygulaması
5. **Regresyon Örneği**: Sürekli değer tahminleme

## 🔗 Bağlantılar

- **Kaggle Notebook**: [Decision Trees Türkçe Rehber](https://www.kaggle.com/code/mustafaatakanyucel/karar-agaclari-decision-trees/notebook)
- **Medium Makalesi**: [Karar Ağaçları - Detaylı Rehber]((https://mustafaatakanyucel.medium.com/hands-on-ml-türkçe-öğrenim-notları-7-056ad61ff1e0))

## 📚 Kaynaklar

- "Hands-On Machine Learning with Scikit-Learn and TensorFlow" - Aurélien Géron
- Scikit-learn Documentation
- Elements of Statistical Learning

## 🤝 Katkı

Bu proje açık kaynak bir öğrenme materyalidir. Katkılarınız için:

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 📧 İletişim

Sorularınız ve önerileriniz için:
- GitHub Issues üzerinden ulaşabilirsiniz
- Medium makalesi altından yorum yapabilirsiniz


⭐ **Beğendiyseniz yıldız vermeyi unutmayın!**

