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
```
kaggle datasets download -d csafrit2/maternal-health-risk-data
unzip maternal-health-risk-data.zip
```

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
data.groupby(by="Age").agg({
    "BS": ["max", "min", "mean", "std"],
    "BodyTemp": ["max", "min", "mean", "std"],
    "HeartRate": ["max", "min", "mean", "std"],
})

Menampilkan grafik BS (Kadar gula darah), BodyTemp (Suhu tubuh), dan HeartRate (Detak Jantung) per grup usia.
![Gula darah terahadap usia](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/Blood%20Sugar%20per%20Age%20Group.JPG) 
![Suhu tubuh terhadap usia](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/Body%20Temp%20per%20Age%20Group.JPG) 
![Detak jantung terhadap usia](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/Heart%20Rate%20per%20Age%20Group.JPG)

Insights:Kadar gula darah didapati rendah pada wanita hamil berusia muda, sedangkan pada usia 35 tahun ke atas, kadar gula darah mulai meningkat. Hal ini juga menyebabkan klasifikasi resiko tinggi (high risk). Suhu tubuh dan usia tidak terlalu berkorelasi. Detak jantung dan usia juga tidak terlalu berkorelasi.

Menganalisa tingkat resiko per fitur (Usia, Tekanan darah, Kadar gula darah, Suhu tubuh, dan Detak jantung)
data.groupby(by="RiskLevel").agg({
    "Age": ["max", "min", "mean", "std"],
    "SystolicBP": ["max", "min", "mean", "std"],
    "DiastolicBP": ["max", "min", "mean", "std"],
    "BS": ["max", "min", "mean", "std"],
    "BodyTemp": ["max", "min", "mean", "std"],
    "HeartRate": ["max", "min", "mean", "std"],
})

Menampilkan grafik karakteristik terhadap tingkat resiko:
![Risk Level terhadap Age](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/Risk%20Level%20terhadap%20Age.JPG) 
![Risk Level terhadap tekanan darah sistol](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/Risk%20Level%20terhadap%20SystolicBP.JPG) 
![Risk Level terhadap tekanan darah diastol](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/Risk%20Level%20terhadap%20DiastolicBP.JPG)
![Risk Level terhadap kadar gula darah](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/Risk%20Level%20terhadap%20BS.JPG) 
![Risk Level terhadap detak jantung](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/Risk%20Level%20terhadap%20Heart%20Rate.JPG) 

Mencari korelasi antara usia, tekanan darah, kadar gula darah dan detak jantung
    correlation = data[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'HeartRate', 'RiskLevel']].corr(numeric_only=True)
    
Memetakan korelasi dalam bentuk heatmap. 
![Correlation Matrix Heatmap](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/Correlation%20Matrix.JPG)

Memetakan korelasi dalam bentuk scatterplot. 


## **Data Preparation**
### Melihat kembali fitur data
data.head()

len(data)

sns.set_palette("Dark2")
sns.boxplot(data)
![Fitur](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/Fitur.JPG)

Melihat lebih lanjut data dengan kecurigaan outliers
![detail outliers](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/detail.JPG)

### Kesan
Data yang digunakan berasal dari Kaggle: Maternal Health Risk Data https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data

Data yang akan digunakan terdiri dari 1014 baris dan 7 kolom. Kolom "RiskLevel" akan menjadi target (y) sedangkan kolom "Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", dan "HeartRate" akan menjadi fitur penentu faktor resiko (X).

Kondisi data:
- Kolom "RiskLevel" sudah berisikan data numerik hasil transformasi data kategorikal 'Low Risk', 'Mid Risk', 'High Risk'.
- Kolom "BodyTemp" sudah diubah ke satuan Celcius dari data awal dalam satuan Fahrenheit untuk memudahkan visualisasi dan komunikasi karena di Indonesia lebih lazim menggunakan satuan Celcius.
- Dari visualisasi data, terlihat bahwa fitur-fitur yang ada memiliki skala variabel yang berbeda, sehingga harus dilakukan scaling sebelum membagi data menjadi train dan test.
- Pada kolom "Age", "BS", dan "BodyTemp", terlihat ada beberapa data yang di luar dari boxplot. Akan tetapi, justru ini merupakan data yang penting karena berkaitan dengan kesehatan manusia sehingga tidak dihapus.

Fitur dalam data:
- Age: Usia saat wanita mengandung (dalam satuan tahun).
- SystolicBP: Nilai atas dari tekanan darah (dalam satuan mmHg).
- DiastolicBP: Nilai bawah dari tekanan darah (dalam satuan mmHg).
- BS: Kadar gula darah dalam satuan konsentrasi molar (mmol/L).
- BodyTemp: Suhu tubuh dalam satuan derajat Celcius (C).
- HeartRate: Detak jantung normal pada posisi istirahat (dalam satuan beats per minute).
- RiskLevel: Tingkat resiko terprediksi selama kehamilan (dalam satuan numerik --> low risk=0, mid risk=1, high risk=2).

data.RiskLevel.value_counts()
> Dari dataset terdapat: Low risk = 406 data, Mid risk = 336 data, High risk = 272 data

### Preparation 1: Membagi Data menjadi Fitur x dan y
x adalah fitur usia, tekanan darah, kadar gula darah, suhu tubuh, dan detak jantung.
y adalah fitur target, yaitu tingkat resiko kehamilan.

X = data.drop(["RiskLevel"], axis =1)
y = data["RiskLevel"]

### Preparation 2: Scaling
Scaling membantu membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma sehingga hasil model akan lebih baik. Langkah selanjutnya adalah melakukan standarisasi pada fitur numerik yaitu 'Age', 'SystolicBP', 'DiastolicBP', 'BS', 'HeartRate' dengan menggunakan teknik StandardScaler dari library Scikitlearn. Hal ini dilakukan karena skala variabel yang berbeda-beda dari setiap fitur yang ada dalam dataset.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

### Train-Test-Split
Membagi dataset menjadi data latih (train) dan data uji (test) dengan perbandingan 70:30. Ratio ini dianggap cukup karena besar data yang adalah 1014. Ratio ini juga diambil dengan pengujian pada ratio 80:20 dimana hasil akurasi rendah lalu diulang dengan 70:30. Selain itu, ratio ini memberikan nilai accuracy yang lebih baik pada data test, dikarenakan jumlah data test cukup besar dan model tidak overfitting.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=101, stratify=y)

Total # of sample in whole dataset: 1014
Total # of sample in train dataset: 709
Total # of sample in test dataset: 305


## **Model Development**
Tahap selanjutnya: mengembangkan model machine learning dengan empat algoritma. Kemudian, kita akan mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Keempat algoritma yang akan kita gunakan, antara lain:

### 1. K-Nearest Neighbor.
   KNN menggunakan algoritma untuk mencari ‘kesamaan fitur’ untuk memprediksi nilai dari data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan.

   Untuk kasus klasifikasi seperti model yang akan kita bangun, data train yang terdiri dari beberapa fitur dengan label yang sudah diketahui (low risk, mid risk, high risk) akan digunakan sebagai acuan untuk mengklasifikasikan fitur-fitur baru. Data baru akan dianggap sebagai titik baru.

   KNN akan mencari k fitur lain dalam data train yang paling mirip dengan fitur baru kita. Kemiripan ini biasanya diukur berdasarkan jarak (misalnya, jarak Euclidean). Nilai k adalah sebuah bilangan bulat positif yang kita tentukan sebelumnya. Setelah menemukan k fitur terdekat, kita akan melihat label (low risk, mid risk, high risk) dari masing-masing fitur tersebut. Fitur baru kita akan diberi label yang paling sering muncul di antara k fitur terdekat tersebut. Misalnya, jika dari 5 fitur terdekat, 3 fitur adalah high risk dan 2 fitur adalah mid risk, maka fitur baru kita akan diklasifikasikan sebagai high risk.

   Keuntungan menggunakan KNN: KNN tidak membuat asumsi tentang distribusi data, sehingga cocok untuk berbagai jenis data. Algoritma KNN juga relatif sederhana untuk diimplementasikan karena konsep mencari tetangga terdekat sangat mudah dipahami dan divisualisasikan.

   Akan tetapi, ada beberapa hal yang perlu diperhatikan dalam menggunakan KNN, yaitu nilai k yang tepat sangat berpengaruh pada hasil klasifikasi. Nilai k yang terlalu kecil bisa membuat model terlalu sensitif terhadap noise, sedangkan nilai k yang terlalu besar bisa membuat batas keputusan menjadi terlalu kabur. Pilihan metrik jarak juga penting. Jarak Euclidean adalah yang paling umum digunakan, tetapi ada juga metrik jarak lainnya seperti Manhattan distance atau Minkowski distance. Jika fitur-fitur memiliki skala yang sangat berbeda, sebaiknya dilakukan normalisasi terlebih dahulu agar fitur-fitur tersebut memiliki kontribusi yang sama dalam perhitungan jarak.

   Create and fit a KNN model dengan nilai k=1
   knn = KNeighborsClassifier(n_neighbors=1)
   knn.fit(X_train, y_train)

   Make predictions on the test data
   knn_preds = knn.predict(X_test)

   Evaluasi nilai k=1
   print('K Nearest Neighbors K=1')
   print('\n')
   print(confusion_matrix(y_test,knn_preds))
   print('\n')
   print(classification_report(y_test,knn_preds))
   ![Hasil evaluasi knn k=1](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/knn%20k1.JPG)

   Mencari nilai error pada rentang k dari 1-20
   error_rate = []
   for i in range(1,20):
      knn = KNeighborsClassifier(n_neighbors=i)
      knn.fit(X_train,y_train)
      pred_i = knn.predict(X_test)
      error_rate.append(np.mean(pred_i != y_test))
   ![error rate vs k value](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/error%20vs%20k%20value.JPG)

   Insight: Error terendah ada pada nilai K = 1

   Membangun model KNN dengan 1 neighbor berdasarkan evaluasi di atas.
   y_pred_train_knn = knn.predict(X_train)
   y_pred_test_knn = knn.predict(X_test)

   ### Evaluasi akurasi model KNN
accuracy_train_knn = accuracy_score(y_pred_train_knn, y_train)
accuracy_test_knn = accuracy_score(y_pred_test_knn, y_test)
--> Hasil: KNN - accuracy_train: 0.9111424541607899, KNN - accuracy_test: 0.7934426229508197
--> Insight: Dengan KNN nilai accuracy pada data train = 91% dan data test = 79%

### 2. Random Forest.
   Random forest adalah algoritma yang kuat dan fleksibel yang dapat digunakan untuk berbagai masalah machine learning. Algoritma random forest adalah salah satu algoritma supervised learning. Ia dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi.

  Random forest merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Random forest menggabungkan banyak model sederhana ini menjadi satu model yang lebih kuat. Model-model sederhana yang digunakan dalam random forest biasanya adalah Decision Tree. Setiap Decision Tree dalam random forest dibangun dengan cara yang sedikit berbeda. Misalnya, setiap pohon hanya menggunakan sebagian data dan sebagian fitur secara acak. Hal ini membuat setiap pohon memiliki "pandangan" yang sedikit berbeda tentang data, sehingga mengurangi risiko overfitting (model terlalu cocok dengan data pelatihan). Setelah setiap pohon keputusan memberikan prediksinya, hasil prediksi dari semua pohon ini akan dihitung. Prediksi yang paling sering muncul akan menjadi prediksi akhir dari random forest.

  Keuntungan dari Random Forest adalah menghasilkan prediksi yang lebih akurat dibandingkan dengan model tunggal, karena menggabungkan banyak decision tree. Model random forest cenderung lebih stabil dan tidak mudah overfitting. Hal ini karena setiap pohon dalam hutan dibangun dengan data yang sedikit berbeda dan fitur yang dipilih secara acak. Random forest dapat bekerja dengan baik pada data numerik maupun kategorikal, serta dapat menangani data yang memiliki missing values. Random forest dapat memberikan informasi tentang pentingnya setiap fitur dalam memprediksi target variabel, serta dapat diterapkan pada dataset yang besar dan kompleks.

  Akan tetapi, karena melibatkan banyak decision tree, random forest dapat membutuhkan waktu komputasi yang lebih lama dibandingkan dengan algoritma yang lebih sederhana, terutama untuk dataset yang besar. Model random forest dapat menjadi sangat kompleks, sehingga sulit untuk diinterpretasi secara mendalam. Selain itu, parameter seperti jumlah pohon, kedalaman pohon, dan jumlah fitur yang dipilih secara acak perlu diset dengan tepat untuk mendapatkan hasil yang optimal.

  Menentukan class weight
  Karena distribusi data tidak merata pada 3 kelompok resiko yakni low risk sebesar .. data, mid risk sebesar .. data, dan high risk sebesar .. data, maka perlu ditentukan class weight agar model lebih seimbang dalam menentukan klasifikasi. Dengan memberikan bobot yang lebih tinggi pada kelas minoritas, random forest akan lebih fokus pada data yang jarang muncul. Ini akan meningkatkan kemampuan model dalam mengklasifikasikan data dari kelas minoritas dengan benar. Dengan menggunakan class weight, kita dapat mengurangi bias ini dan membuat model lebih adil.

  Penggunaan class weight umumnya lazim pada klasifikasi bidang kesehatan dimana kemunculan penyakit lebih jarang daripada kondisi normal. Karena itu pula, pada kasus tingkat resiko kehamilan ini, bobot lebih kecil diberikan pada tingkat low risk, dan bobot lebih besar diberikan pada tingkat mid risk dan high risk.
  
  class_weight = {0: 0.2, 1: 0.4, 2: 0.4}
  
  Membangun model Random Forest dengan hyperparameter grid search CV
  - Impor library yang diperlukan yakni RandomForestClassifier dan GridSearchCV
  - Mengimplementasikan class weight
  - Membuat grid parameter dimana ditentukan n-estimators (jumlah decision tree dalam forest yang dibuat), criterion (kriteria splitting dipilih Gini dan entropy), depth (kedalaman setiap decision tree), dan min_samples_leaf (jumlah minimal sampel dalam setiap leaf node).
  - Membangun GridSearchCV object
  - Fit GridSearch kepada data train
  - Menentukan best model Random Forest yang selanjutnya akan digunakan

  Impor library yang dibutuhkan
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import GridSearchCV

  create a Random Forest Classifier
  forest = RandomForestClassifier(class_weight=class_weight)

  define the hyperparameter grid
  param_grid = {
    'n_estimators': [100, 300, 500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [20, 25, 30],
    'min_samples_leaf': [2, 3, 5]
  }

  create the GridSearchCV object
  grid_search_forest = GridSearchCV(forest, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

  fit the grid search to the data
  grid_search_forest.fit(X_train, y_train)

  print the best parameters and the corresponding accuracy
  print('Best Parameters: ', grid_search_forest.best_params_)
  print('Best Accuracy: ', grid_search_forest.best_score_)

  get the best model
  best_forest = grid_search_forest.best_estimator_

  Hasil: 
  Best Parameters:  {'criterion': 'gini', 'max_depth': 25, 'min_samples_leaf': 2, 'n_estimators': 300}
  Best Accuracy:  0.8152731994805714

  Menggunakan best model untuk membuat prediksi pada data train dan test
  y_pred_train_rf = grid_search_forest.predict(X_train)
  y_pred_test_rf = grid_search_forest.predict(X_test)

  ### Evaluasi akurasi model RF
  accuracy_train_rf = accuracy_score(y_pred_train_rf, y_train)
  accuracy_test_rf = accuracy_score(y_pred_test_rf, y_test)
  --> Hasil: Random Forest - accuracy_train: 0.9167842031029619, Random Forest - accuracy_test: 0.7836065573770492
  --> Insight: Dengan Random Forest, nilai accuracy pada data train = 91% dan data test = 77%
   
### 3. Decision Tree Classifier.
  Decision tree adalah algoritma yang membuat keputusan dengan membagi data ke subset berdasarkan fitur yang ada. Sama seperti sebuah pohon, node/persimpangan melambangkan keputusan, cabang melambangkan hasil dan daun (leaf node) melambangkan prediksi akhir (class label).

  Keuntungan Decision Tree adalah mudah dimengerti dan divisualisasikan sehingga cocok untuk menjelaskan proses membuat keputusan. Decision Tree juga mampu menangkap hubungan non linear antara variabel fitur dan target. Decision Tree juga dapat diaplikasikan pada data numerik maupun kategorikal.

  Kelemahan Decision Tree adalah mudah menjadi overfitting sehingga prediksi jadi tidak akurat, serta ketidakstabilan jika ada perubahan data. Perubahan data sekecil apapun akan membawa kepada perubahan signifikan pada struktur tree.

   from sklearn.tree import DecisionTreeClassifier
   decision_tree = DecisionTreeClassifier()
   decision_tree.fit(X_train, y_train)
   y_pred_train_dt = decision_tree.predict(X_train)
   y_pred_test_dt = decision_tree.predict(X_test)

  ### Evaluasi akurasi model Decision Tree
  accuracy_train_dt = accuracy_score(y_pred_train_dt, y_train)
  accuracy_test_dt = accuracy_score(y_pred_test_dt, y_test)
  --> Hasil: Decision Tree - accuracy_train: 0.9308885754583921, Decision Tree - accuracy_test: 0.7967213114754098
  --> Insight: Dengan Decision Tree, nilai accuracy pada data train = 93% dan data test = 79%
   
### 4. XGBoost Classifier.
  XGB atau disebut juga extreme gradient boosting menggunakan decision tree sebagai dasar dari modelnya. XGBoost menggunakan framework gradient boosting dimana setiap tree baru dilatih untuk mengoreksi error pada tree sebelumnya. Proses iteratif ini akan meningkatkan performa secara keseluruhan. XGBoost juga menggunakan teknik regularisasi untuk mencegah overfitting. XGBoost juga sangat scalable serta memiliki mekanisme untuk mengatasi permasalahan missing values selama training.

  Kelemahan dari XGBoost adalah kompleksitas dan kesulitan untuk menginterpretasikan keputusan/prediksi yang dibuat oleh model.

  - Instalasi xgboost
  - Menggunakan GridSearchCV untuk menguji performa
  - Membangun XGB Classifier dengan jumlah estimator awal 50
  - Menentukan beberapa parameter yaitu eta (learning rate), max_depth (kedalaman pada tiap decision tree), dan min_child_weight (jumlah minimal children nodes yang digunakan dalam partisi sebelum membentuk leaf node).
  - Menggunakan GridSearchCV untuk hyperparameter tuning.
  - Fitting objek GridSearchCV pada data training (X dan y).
    
  !pip install xgboost
  from sklearn.model_selection import GridSearchCV
  import xgboost as xgb
  xgb_model = xgb.XGBClassifier(n_estimators=50)

  Define hyperparameters to tune
  param_grid = {
    'eta': [0.01, 0.05, 0.1, 0.35],
    'max_depth': [2, 4, 7, 9, 12, 17],
    'min_child_weight': [2, 4, 7, 9, 12, 17]
  }

  Perform Grid Search Cross Validation to find the best hyperparameters
   xgb_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3)
   xgb_search.fit(X_train, y_train)

  Get the best model from Grid Search
   xgb_classifier = xgb_search.best_estimator_

  Melatih model dengan data training
   xgb_classifier.fit(X_train, y_train)

  Membuat prediksi
   y_pred_train_xgb = xgb_classifier.predict(X_train)
   y_pred_test_xgb = xgb_classifier.predict(X_test)

  ### Evaluasi akurasi model XGB Classifier
   accuracy_train_xgb = accuracy_score(y_pred_train_xgb, y_train)
   accuracy_test_xgb = accuracy_score(y_pred_test_xgb, y_test)
   --> Hasil: Decision Tree - accuracy_train: 0.9294781382228491, Decision Tree - accuracy_test: 0.7967213114754098
   --> Insight: Dengan XGB Classifier, nilai accuracy pada data train = 93% dan data test = 80%

### Memilih model dengan accuracy terbaik
Dari keempat algoritma model yang ada, dipilih 1 model terbaik.

Define a dictionary to store the models
  models = {
    'KNN': KNeighborsClassifier(n_neighbors=1),
    'Random Forest': RandomForestClassifier(class_weight=class_weight, **grid_search_forest.best_params_),
    'Decision Tree': DecisionTreeClassifier(),
    'XGB Classifier': XGBClassifier(**xgb_search.best_params_)
}

Train and evaluate each model
  best_model = None
  best_accuracy = 0

  for model_name, model in models.items():

Train the model
  model.fit(X_train, y_train)

Make predictions on the test set
  y_pred = model.predict(X_test)

Calculate the accuracy
  accuracy = accuracy_score(y_test, y_pred)

Update the best model if the current model has higher accuracy
  if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_model = model
    best_model_name = model_name

  print(f"The best model is {best_model_name} with an accuracy of {best_accuracy}")
  --> The best model is XGB Classifier with an accuracy of 0.8032786885245902

Insight
Berdasarkan evaluasi di atas, model XGB adalah yang paling akurat dengan nilai 80.33% accuracy.

### Menguji Model
prediksi = X_test[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in models.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)
pd.DataFrame(pred_dict)

--> Keempat algoritma menghasilkan hasil yang sama. 

## **Model Evaluation**
### Evaluasi model 
  Accuracy adalah metrik umum yang digunakan pada kasus klasifikasi. Metrik ini mengukur proporsi keseluruhan dari prediksi benar yang dibuat oleh model.

> Rumus metrik accuracy: Accuracy = (TP + TN) / (TP + TN + FP + FN)

Dimana: TP = true positives TN = true negatives FP = false positives FN = false negatives

  Dengan kata lain metrik accuracy mengukur jumlah prediksi benar (baik positif maupun negatif) dibandingkan total prediksi. Nilai 1 (100%) berarti model berhasil memprediksi seluruh data dengan benar, sedangkan nilai 0 (0%) berarti model gagal memprediksi seluruh data.

  Accuracy sangat mudah dimengerti dan diinterpretasi. Metrik ini juga sangat sering digunakan dan dilaporkan. Kelemahan accuracy adalah jika digunakan pada data yang tidak seimbang, dapat menghasilkan misleading. Hal ini terjadi karena prediksi akan menghasilkan lebih banyak nilai pada data yang lebih banyak, sehingga diukur sebagai akurasi yang tinggi. Metrik ini juga tidak menampilkan informasi mengenai error yang dibuat oleh model.

  Metrik ini dipilih karena dataset memiliki data yang hampir seimbang antara kategori low risk, mid risk, dan high risk. Selain itu, output yang dihasilkan adalah klasifikasi sehingga metrik accuracy tepat digunakan.

### Merangkum hasil accuracy dari 4 model

results_df = pd.DataFrame({
    'Model': ['KNN', 'Random Forest', 'Decision Tree', 'XGB Classifier'],
    'Accuracy Train': [accuracy_train_knn, accuracy_train_rf, accuracy_train_dt, accuracy_train_xgb],
    'Accuracy Test': [accuracy_test_knn, accuracy_test_rf, accuracy_test_dt, accuracy_test_xgb]
})
accuracy = results_df[['Model', 'Accuracy Train', 'Accuracy Test']]

![Accuracy](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/Accuracy.JPG)

### Memetakan accuracy 4 algoritma dalam plot
![Model accuracy comparison](https://github.com/Shue84/Predictive-Analysis/blob/main/gambar/Model%20Accuracy%20Comparison.JPG)

--> Dari hasil di atas, didapatkan bahwa keempat model memiliki nilai akurasi yang hampir sama, sehingga keempat model dapat digunakan untuk klasifikasi tingkat resiko kehamilan, dengan model XGBoost sebagai yang terutama.

## Kesimpulan

1. Usia, tekanan darah, gula darah, suhu tubuh, dan detak jantung ibu merupakan fitur kesehatan yang digunakan untuk menilai tingkat resiko kehamilan. Kelima fitur itu juga disebutkan dalam beberapa artikel (Gloria, 2022 dan Pawestri, 2023) sebagai faktor resiko yang perlu diperhatikan selama kehamilan. Dari penjabaran solution statements dan langkah-langkah yang dilakukan pada proyek ini, hubungan antara fitur saling terkait dan mempengaruhi klasifikasi tingkat resiko kehamilan. Hal ini juga tergambar dari tahapan Exploratory Data Analysis dan Explanatory Data Analysis. 

2. Membangun model machine learning untuk memprediksi tingkat resiko kehamilan dengan memperhatikan 5 fitur, yakni usia (Age), tekanan darah (Systolic dan Diastolic Blood Pressure), kadar gula darah (Blood Sugar), suhu tubuh (Body Temperature), dan detak jantung (Heart Rate). Dengan 4 algoritman yakni KNN, Random Forest, Decision Tree, dan XGBoost, model terbaik dibangun dan dilatih. Keempat algoritma dievaluasi dengan metrik accuracy untuk menilai ketepatan klasifikasi yang dihasilkan model. Nilai accuracy pada keempatnya hampir sama dengan nilai tertingggi pada model XGBoost (80.33%). Model dapat digunakan untuk memprediksi tingkat resiko kehamilan pada wanita dengan mempertimbangkan fitur kesehatannya. 

3. Walau tingkat akurasi model sudah cukup tinggi (di atas 80%) namun sebaiknya model terus dilatih dengan lebih banyak data. Karena itu, pengumpulan data harus terus dilakukan dan data dilatihkan kembali ke model agar tingkat akurasi dapat lebih meningkat dan performa model dapat lebih baik.

## Referensi
1. Gloria, N. (18 Maret 2022). 9 Faktor Kehamilan Resiko Tinggi, Beserta Penanganannya. Viva. https://www.viva.co.id/gaya-hidup/kesehatan-intim/1458611-kehamilan-risiko-tinggi

2. Karnesyia, A. (2 Feb 2024). 13 Tanda Kehamilan Beresiko Tinggi dan Penyebab yang Perlu Bunda Tahu. HaiBunda. https://www.haibunda.com/kehamilan/20240131135720-49-327790/13-tanda-kehamilan-berisiko-tinggi-dan-penyebab-yang-perlu-bunda-tahu

3. Kementerian Kesehatan Republik Indonesia. 2021. Profil Kesehatan Indonesia 2020. https://r.search.yahoo.com/_ylt=AwrKCdroQgNnt6kMfDPLQwx.;_ylu=Y29sbwNzZzMEcG9zAzIEdnRpZAMEc2VjA3Ny/RV=2/RE=1728295784/RO=10/RU=https%3a%2f%2fkemkes.go.id%2fapp_asset%2ffile_content_download%2fProfil-Kesehatan-Indonesia-2020.pdf/RK=2/RS=GY0YoJ2jFSy6ggsJBxiuOEndidg-

4. Pawestri, H. S. (25 Jan 2023). Penyebab Kehamilan Resiko Tinggi dan Cara Menjalaninya. HelloSehat. https://hellosehat.com/kehamilan/kandungan/masalah-kehamilan/apa-itu-kehamilan-risiko-tinggi/


