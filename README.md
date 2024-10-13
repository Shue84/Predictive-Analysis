# **Proyek Predictive Analysis: Maternal Health - Menilai Resiko Kehamilan**

- **Nama:** Suyanti Witono
- **Email:** suyanti.witono@bpkpenaburjakarta.or.id
- **ID Dicoding:** suyanti_witono_tfde


## **Latar Belakang**
Kehamilan adalah hal yang membahagiakan bagi wanita pada khususnya dan bagi keluarga pada umumnya. Akan tetapi, kehamilan juga perlu diperhatikan karena mengandung resiko.

> Jumlah kematian ibu yang dihimpun dari pencatatan program kesehatan keluarga di Kementerian Kesehatan pada tahun 2020 menunjukkan 4.627 kematian di Indonesia. Jumlah ini menunjukkan peningkatan dibandingkan tahun 2019 sebesar 4.221 kematian. Berdasarkan penyebab, sebagian besar kematian ibu pada tahun 2020 disebabkan oleh perdarahan sebanyak 1.330 kasus, hipertensi dalam kehamilan sebanyak 1.110 kasus, dan gangguan sistem peredaran darah sebanyak 230 kasus. (Kemenkes, 2021)

Usia menjadi salah satu penyebab resiko. Kehamilan pada usia di bawah 17 tahun mengandung resiko karena organ reproduksi belum berkembang sempurna. Seringkali pula, wanita di bawah umur ini tidak mencari pertolongan medis selama kehamilan sampai pada saat melahirkan. Kehamilan di usia tua juga beresiko, terutama di atas usia 35 tahun karena meningkatnya komplikasi dan peluang keguguran (Gloria, 2022).  Selain usia, faktor lain seperti tekanan darah (blood pressure), kadar gula dalam darah (blood sugar level), dan gaya hidup juga mempengaruhi resiko kehamilan (Pawestri, 2023).

Demam selama hamil terutama pada trimester pertama dapat menyebabkan masalah pada perkembangan janin (Karnesyia, 2024). Selain itu, kondisi preeklampsia juga dapat menyebabkan kondisi darurat pada ibu hamil dan janin yang dikandung. Preeklampsia diikuti dengan tekanan darah tinggi, adanya protein pada urine, serta dapat memburuk menjadi serangan jantung atau gagal ginjal akut. Karena itu, usia, tekanan darah, dan kadar gula darah penting untuk dipantau.

Predictive analysis ini akan berfokus pada pengolahan data usia, tekanan darah sistol, tekanan darah diastol, kadar gula darah, suhu tubuh, dan detak jantung untuk menentukan tingkat resiko kehamilan. Dataset yang digunakan adalah https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data

### **Referensi**
1. Gloria, N. (18 Maret 2022). *9 Faktor Kehamilan Resiko Tinggi, Beserta Penanganannya*. Viva. https://www.viva.co.id/gaya-hidup/kesehatan-intim/1458611-kehamilan-risiko-tinggi
2. Karnesyia, A. (2 Feb 2024). *13 Tanda Kehamilan Beresiko Tinggi dan Penyebab yang Perlu Bunda Tahu*. HaiBunda. https://www.haibunda.com/kehamilan/20240131135720-49-327790/13-tanda-kehamilan-berisiko-tinggi-dan-penyebab-yang-perlu-bunda-tahu
3. Kementerian Kesehatan Republik Indonesia. 2021. *Profil Kesehatan Indonesia 2020*. https://r.search.yahoo.com/_ylt=AwrKCdroQgNnt6kMfDPLQwx.;_ylu=Y29sbwNzZzMEcG9zAzIEdnRpZAMEc2VjA3Ny/RV=2/RE=1728295784/RO=10/RU=https%3a%2f%2fkemkes.go.id%2fapp_asset%2ffile_content_download%2fProfil-Kesehatan-Indonesia-2020.pdf/RK=2/RS=GY0YoJ2jFSy6ggsJBxiuOEndidg-
4. Pawestri, H. S. (25 Jan 2023). *Penyebab Kehamilan Resiko Tinggi dan Cara Menjalaninya*. HelloSehat. https://hellosehat.com/kehamilan/kandungan/masalah-kehamilan/apa-itu-kehamilan-risiko-tinggi/


## **Business Understanding**

### *Problem Statements:*
1. Bagaimana hubungan antara usia, tekanan darah, gula darah, suhu tubuh, dan detak jantung ibu terhadap resiko kehamilan?
2. Berapa tingkat resiko kehamilan dengan parameter tertentu dari kesehatan ibu?

### *Goals:*
1. Mengetahui korelasi antara usia, tekanan darah, gula darah, suhu tubuh, dan detak jantung terhadap resiko kehamilan.
2. Membuat model machine learning yang dapat memprediksi resiko kehamilan berdasarkan parameter tertentu dari kesehatan ibu.

### *Solution Statements:*
Langkah awal adalah mengetahui korelasi antara usia, tekanan darah, gula darah, suhu tubuh, dan detak jantung terhadap resiko kehamilan dengan cara menganalisis dataset yang tersedia.

Pertama, kolom tingkat resiko kehamilan harus diubah menjadi data numerik sehingga memudahkan untuk analisis. Perubahan tersebut mengikuti setting berikut: low risk = 0 mid risk = 1 high risk = 2

Data dicek lalu dianalisis dengan teknik Exploratory Data Analysis (EDA) bivariate yaitu membandingkan masing-masing fitur terhadap tingkat resiko kehamilan. Analisis data juga akan ditampilkan dalam bentuk diagram sehingga dapat tervisualisasi dengan baik.

Setelah itu, EDA dilakukan untuk melihat hubungan antara usia ("Age) dengan fitur lainnya seperti tekanan darah ("SistolicBP" dan "DiastolicBP), kadar gula darah ("BS"), suhu tubuh ("BodyTemp"), dan detak jantung ("HeartRate").

Metode korelasi juga dilakukan untuk melihat hubungan antar 7 fitur (kolom) yang ada pada dataset. Visualisasi dilakukan dengan menampilkan heatmap serta scatterplot. Hal ini dapat menjawab pertanyaan no 1 pada Problem Statements.

Untuk menjawab Problem Statements kedua, maka dibangun model dengan outcomes yang diharapkan adalah 3 kategori: high risk, mid risk, low risk. Data dibagi menjadi data train dan test dengan rasio 70:30 (dilakukan percobaan untuk rasio standar 80:20 kemudian dicek setelah dijalankan apakah memenuhi harapan atau perlu dilakukan penyesuaian untuk meningkatkan accuracy).

Model machine learning akan dibangun dengan beberapa algoritma dan dipilih model dengan kesalahan prediksi terkecil.
1. K-Nearest Neighbor.
2. Random Forest.
3. Decision Tree Classifier.
4. XGB Classifier

### *Metrik Evaluasi*
Metrik yang digunakan adalah Accuracy (ketepatan klasifikasi tingkat resiko kehamilan - high risk, mid risk, low risk).

Model lalu disusun berdasarkan evaluasi dengan accuracy pada train dan test dan algoritma yang memberikan hasil accuracy tertinggi. Visualisasi perbandingan 4 algoritma juga akan ditunjukkan dalam bentuk barplot graph.

Di akhir, dilakukan ujicoba model apakah memberikan hasil prediksi sesuai dengan yang diharapkan.


## **Data Understanding**
Data yang digunakan berasal dari Kaggle: Maternal Health Risk Data https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data
Dataset terdiri dari 1014 data kesehatan ibu hamil dengan parameter kesehatan yaitu usia, tekanan darah sistol, tekanan darah diastol, kadar gula darah, suhu tubuh, dan detak jantung. Tingkat resiko (high risk, mid risk, low risk) adalah fitur target.

### Impor Library yang dibutuhkan
numpy
pandas
matplotlib.pyplot
seaborn

### Download file dari Kaggle
kaggle datasets download -d csafrit2/maternal-health-risk-data
unzip maternal-health-risk-data.zip

### Membaca data pada file yang sudah di-download
data = pd.read_csv("/content/Maternal Health Risk Data Set.csv")

### Keterangan dari tabel dataset
- Age: Usia saat wanita mengandung (dalam satuan tahun).
- SystolicBP: Nilai atas dari tekanan darah (dalam satuan mmHg).
- DiastolicBP: Nilai bawah dari tekanan darah (dalam satuan mmHg).
- BS: Kadar gula darah dalam satuan konsentrasi molar (mmol/L).
- BodyTemp: Suhu tubuh dalam satuan Fahrenheit (F).
- HeartRate: Detak jantung normal pada posisi istirahat (dalam satuan beats per minute).
- RiskLevel: Tingkat resiko terprediksi selama kehamilan dengan pertimbangan atribut sebelumnya attribute.

### Mengetahui lebih lanjut mengenai informasi data yang tersedia:
- Memastikan tidak ada data yang kosong
  data.isna().sum()
- Mengetahui dan menghilangkan data duplikat
  data[data.duplicated(keep='first')]
- Mengubah data 'RiskLevel' menjadi numerik untuk memudahkan pengolahan statistik lebih lanjut dan plot graph.
  data.replace({"high risk":2, "mid risk":1, "low risk":0}, inplace=True)
- Mengubah data suhu tubuh dari Fahrenheit ke Celcius
  data['BodyTemp'] = (data['BodyTemp'] - 32) * 5 / 9
- Memastikan data sudah sesuai
  data.describe()

### Exploratory Data Analysis (EDA)
Bivariate Analysis - Usia terhadap Tingkat Resiko
    Menggunakan boxplot 
    ![Usia terhadap tingkat resiko](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/Age%20by%20Risk%20Level.JPG)
    
Bivariate Analysis - Tekanan Darah terhadap Tingkat Resiko
    Memetakan distribusi systolicBP dan diastolicBP dengan histogram
    Menggunakan boxplot
    ![Tekanan darah distribusi](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/Blood%20Pressure%20Distribution.JPG)
    ![Tekanan darah terhadap tingkat resiko](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/Blood%20Pressure%20by%20Risk%20Level.JPG)
    
Bivariate Analysis - Kadar Gula Darah terhadap Tingkat Resiko
    Menggunakan boxplot
    ![Gula darah terhadap tingkat resiko](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/Blood%20Sugar%20by%20Risk%20Level.JPG) 
    
Bivariate Analysis - Suhu Tubuh terhadap Tingkat Resiko
    Menggunakan violinplot agar lebih terlihat jelas
    ![Suhu tubuh terhadap tingkat resiko](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/Blood%20Temperature%20by%20Risk%20Level.JPG)
    
Bivariate Analysis - Detak Jantung terhadap Tingkat Resiko
    Menggunakan boxplot
    ![Detak jantung terhadap tingkat resiko](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/Heart%20Rate%20by%20Risk%20Level.JPG)

*Insights:*
- Kehamilan dengan resiko rendah (low risk) adalah yang paling sering dengan lebih dari setengah kasus.
- Wanita yang lebih mudah cenderung memiliki kehamilan dengan resiko rendah (low risk) maupun sedang (mid risk), sementara pada wanita berusia di atas 35 tahun, lebih sering diklasifikasikan sebagai resiko tinggi (high risk).
- Wanita hamil yang memiliki kadar gula darah di atas 8 mmol/L, diklasifikasikan sebagai kehamilan dengan resiko tinggi (high risk).
- Distribusi dari nilai bawah pada tekanan darah (diastol) lebih tersebar, antara 60-100 mmHg, dibandingkan dengan distribusi pada nilai atas (sistol), yang cenderung berkisar pada 120 mmHg.
- Tekanan darah tinggi (both sistol dan diastol) serta suhu tubuh yang tinggi diklasifikasikan sebagai kehamilan dengan resiko tinggi (high risk).
- Detak jantung dari wanita hamil lebih terdistribusi normal dan hanya sedikit berhubungan dengan tingkat resiko.

Menganalisis lebih lanjut karakteristik Systolic BP (tekanan darah sistol) dan Diastolic BP (tekanan darah diastol) per grup usia.
data.groupby(by="Age").agg({
    "SystolicBP": ["max", "min", "mean", "std"],
    "DiastolicBP": ["max", "min", "mean", "std"],
})

Menampilkan grafik rata-rata tekanan darah terhadap usia.
![Tekanan darah terhadap usia](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/Blood%20Pressure%20by%20Age%20Group.JPG)
Insight: Berdasarkan usia dan tekanan darah, tekanan darah sangat rendah (baik sistol dan diastol) terekam pada wanita berusia muda, namun tekanan darah normal dan tinggi nampak tidak terlalu berkorelasi dengan usia.

Menganalisis lebih lanjut karakteristik BS (Kadar gula darah), BodyTemp (Suhu tubuh), dan HeartRate (Detak Jantung) per grup usia.
Menampilkan grafik BS (Kadar gula darah), BodyTemp (Suhu tubuh), dan HeartRate (Detak Jantung) per grup usia.
Insights:Kadar gula darah didapati rendah pada wanita hamil berusia muda, sedangkan pada usia 35 tahun ke atas, kadar gula darah mulai meningkat. Hal ini juga menyebabkan klasifikasi resiko tinggi (high risk). Suhu tubuh dan usia tidak terlalu berkorelasi. Detak jantung dan usia juga tidak terlalu berkorelasi.

Menganalisa tingkat resiko per fitur (Usia, Tekanan darah, Kadar gula darah, Suhu tubuh, dan Detak jantung)
Menampilkan grafik karakteristik terhadap tingkat resiko:

Mencari korelasi antara usia, tekanan darah, kadar gula darah dan detak jantung
    correlation = data[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'HeartRate', 'RiskLevel']].corr(numeric_only=True)
Memetakan korelasi dalam bentuk heatmap. 
Memetakan korelasi dalam bentuk scatterplot. 


## **Data Preparation**
### Membagi Data menjadi Fitur x dan y
x adalah fitur usia, tekanan darah, kadar gula darah, suhu tubuh, dan detak jantung.
y adalah fitur target, yaitu tingkat resiko kehamilan.

X = data.drop(["RiskLevel"], axis =1)
y = data["RiskLevel"]

### Scaling
Scaling membantu membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma sehingga hasil model akan lebih baik. Langkah selanjutnya adalah melakukan standarisasi pada fitur numerik yaitu 'Age', 'SystolicBP', 'DiastolicBP', 'BS', 'HeartRate' dengan menggunakan teknik StandardScaler dari library Scikitlearn.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

### Train-Test-Split
Membagi dataset menjadi data latih (train) dan data uji (test) dengan perbandingan 70:30. Ratio ini dianggap cukup karena besar data yang adalah 1014. Ratio ini juga diambil dengan pengujian pada ratio 80:20 dimana hasil akurasi rendah lalu diulang dengan 70:30. 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=101, stratify=y)

Total # of sample in whole dataset: 1014
Total # of sample in train dataset: 709
Total # of sample in test dataset: 305


## **Model Development**
Tahap selanjutnya: mengembangkan model machine learning dengan empat algoritma. Kemudian, kita akan mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Keempat algoritma yang akan kita gunakan, antara lain:
1. K-Nearest Neighbor.
   Pertama menggunakan 1 neighbor untuk melihat tingkat akurasi. Kemudian menguji nilai K yang cocok. Dari hasil tersebut, didapatkan bahwa error terkecil pada nilai K=1. 
   knn = KNeighborsClassifier(n_neighbors=1)
   knn.fit(X_train, y_train)
   y_pred_train_knn = knn.predict(X_train)
   y_pred_test_knn = knn.predict(X_test)

2. Random Forest.
   Menggunakan Hyperparameter Grid Search dan mencari parameter yang sesuai. Didapatkan hasil Best Parameters:  {'criterion': 'gini', 'max_depth': 20, 'min_samples_leaf': 2, 'n_estimators': 500}.
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import GridSearchCV
  forest = RandomForestClassifier(class_weight=class_weight)
  param_grid = {
    'n_estimators': [100, 300, 500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [20, 25, 30],
    'min_samples_leaf': [2, 3, 5] 
  }
  grid_search_forest = GridSearchCV(forest, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
  grid_search_forest.fit(X_train, y_train)
  print('Best Parameters: ', grid_search_forest.best_params_)
  print('Best Accuracy: ', grid_search_forest.best_score_)
  best_forest = grid_search_forest.best_estimator_
  y_pred_train_rf = grid_search_forest.predict(X_train)
  y_pred_test_rf = grid_search_forest.predict(X_test)
   
3. Decision Tree Classifier.
   from sklearn.tree import DecisionTreeClassifier
   decision_tree = DecisionTreeClassifier()
   decision_tree.fit(X_train, y_train)
   y_pred_train_dt = decision_tree.predict(X_train)
   y_pred_test_dt = decision_tree.predict(X_test)
   
4. XGB Classifier.
   !pip install xgboost
   from sklearn.model_selection import GridSearchCV
   import xgboost as xgb
   xgb_model = xgb.XGBClassifier(n_estimators=50)
   param_grid = {
    'eta': [0.01, 0.05, 0.1, 0.35],
    'max_depth': [2, 4, 7, 9, 12, 17],
    'min_child_weight': [2, 4, 7, 9, 12, 17]
    }
   xgb_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3)
   xgb_search.fit(X_train, y_train)
   xgb_classifier = xgb_search.best_estimator_
   xgb_classifier.fit(X_train, y_train)
   y_pred_train_xgb = xgb_classifier.predict(X_train)
   y_pred_test_xgb = xgb_classifier.predict(X_test)

## **Model Evaluation**
### Evaluasi model 
Menggunakan metrik "accuracy" dengan accuracy_score antara train dan test
from sklearn.metrics import accuracy_score, precision_score

### Evaluasi akurasi model KNN
accuracy_train_knn = accuracy_score(y_pred_train_knn, y_train)
accuracy_test_knn = accuracy_score(y_pred_test_knn, y_test)
--> Hasil: KNN - accuracy_train: 0.9111424541607899, KNN - accuracy_test: 0.7934426229508197
--> Insight: Dengan KNN nilai accuracy pada data train = 91% dan data test = 79%

### Evaluasi akurasi model RF
accuracy_train_rf = accuracy_score(y_pred_train_rf, y_train)
accuracy_test_rf = accuracy_score(y_pred_test_rf, y_test)
--> Hasil: Random Forest - accuracy_train: 0.9167842031029619, Random Forest - accuracy_test: 0.7836065573770492
--> Insight: Dengan Random Forest, nilai accuracy pada data train = 92% dan data test = 78%

### Evaluasi akurasi model Decision Tree
accuracy_train_dt = accuracy_score(y_pred_train_dt, y_train)
accuracy_test_dt = accuracy_score(y_pred_test_dt, y_test)
--> Hasil: Decision Tree - accuracy_train: 0.9308885754583921, Decision Tree - accuracy_test: 0.7967213114754098
--> Insight: Dengan Decision Tree, nilai accuracy pada data train = 93% dan data test = 80%

### Evaluasi akurasi model XGB Classifier
accuracy_train_xgb = accuracy_score(y_pred_train_xgb, y_train)
accuracy_test_xgb = accuracy_score(y_pred_test_xgb, y_test)
--> Hasil: Decision Tree - accuracy_train: 0.9294781382228491, Decision Tree - accuracy_test: 0.7967213114754098
--> Insight: Dengan XGB Classifier, nilai accuracy pada data train = 93% dan data test = 80%

### Memetakan accuracy 4 algoritma dalam plot
results_df = pd.DataFrame({
    'Model': ['KNN', 'Random Forest', 'Decision Tree', 'XGB Classifier'],
    'Accuracy Train': [accuracy_train_knn, accuracy_train_rf, accuracy_train_dt, accuracy_train_xgb],
    'Accuracy Test': [accuracy_test_knn, accuracy_test_rf, accuracy_test_dt, accuracy_test_xgb]
})
accuracy = results_df[['Model', 'Accuracy Train', 'Accuracy Test']]

Hasilnya kemudian dipetakan dalam barplot. 
--> Dari hasil di atas, didapatkan bahwa keempat model memiliki nilai akurasi yang hampir sama.

### Mengembangkan model 
models = {
    'KNN': KNeighborsClassifier(n_neighbors=1),
    'Random Forest': RandomForestClassifier(class_weight=class_weight, **grid_search_forest.best_params_),
    'Decision Tree': DecisionTreeClassifier(),
    'XGB Classifier': XGBClassifier(**xgb_search.best_params_)
}

Melatih model: 
best_model = None
best_accuracy = 0

for model_name, model in models.items():
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Update the best model if the current model has higher accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = model_name

The best model is XGB Classifier with an accuracy of 0.8032786885245902

### Menguji Model
prediksi = X_test[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in models.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)

Keempat algoritma menghasilkan hasil yang sama. 
