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
Outcomes yang diharapkan adalah 3 kategori: high risk, mid risk, low risk.
Model machine learning akan dibangun dengan beberapa algoritma dan dipilih model dengan kesalahan prediksi terkecil.
1. Multinominal logistic regression.
2. K-Nearest Neighbor.
3. Random Forest.
4. Decision Tree Classifier.

### *Metrik Evaluasi*
Metrik yang digunakan adalah Mean Squared Error (menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi).


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
Bivariate Analysis - Tekanan Darah terhadap Tingkat Resiko
    Memetakan distribusi systolicBP dan diastolicBP dengan histogram
    Menggunakan boxplot
Bivariate Analysis - Kadar Gula Darah terhadap Tingkat Resiko
    Menggunakan boxplot
Bivariate Analysis - Suhu Tubuh terhadap Tingkat Resiko
    Menggunakan violinplot agar lebih terlihat jelas
Bivariate Analysis - Detak Jantung terhadap Tingkat Resiko
    Menggunakan boxplot

*Insights:*
- Kehamilan dengan resiko rendah (low risk) adalah yang paling sering dengan lebih dari setengah kasus.
- Wanita yang lebih mudah cenderung memiliki kehamilan dengan resiko rendah (low risk) maupun sedang (mid risk), sementara pada wanita berusia di atas 35 tahun, lebih sering diklasifikasikan sebagai resiko tinggi (high risk).
- Wanita hamil yang memiliki kadar gula darah di atas 8 mmol/L, diklasifikasikan sebagai kehamilan dengan resiko tinggi (high risk).
- Distribusi dari nilai bawah pada tekanan darah (diastol) lebih tersebar, antara 60-100 mmHg, dibandingkan dengan distribusi pada nilai atas (sistol), yang cenderung berkisar pada 120 mmHg.
- Tekanan darah tinggi (both sistol dan diastol) serta suhu tubuh yang tinggi diklasifikasikan sebagai kehamilan dengan resiko tinggi (high risk).
- Detak jantung dari wanita hamil lebih terdistribusi normal dan hanya sedikit berhubungan dengan tingkat resiko.

Menganalisis lebih lanjut karakteristik Systolic BP (tekanan darah sistol) dan Diastolic BP (tekanan darah diastol) per grup usia.
Menampilkan grafik rata-rata tekanan darah terhadap usia.
Insight: Berdasarkan usia dan tekanan darah, tekanan darah sangat rendah (baik sistol dan diastol) terekam pada wanita berusia muda, namun tekanan darah normal dan tinggi nampak tidak terlalu berkorelasi dengan usia.

Menganalisis lebih lanjut karakteristik BS (Kadar gula darah), BodyTemp (Suhu tubuh), dan HeartRate (Detak Jantung) per grup usia.
Menampilkan grafik BS (Kadar gula darah), BodyTemp (Suhu tubuh), dan HeartRate (Detak Jantung) per grup usia.
Insights:Kadar gula darah didapati rendah pada wanita hamil berusia muda, sedangkan pada usia 35 tahun ke atas, kadar gula darah mulai meningkat. Hal ini juga menyebabkan klasifikasi resiko tinggi (high risk). Suhu tubuh dan usia tidak terlalu berkorelasi. Detak jantung dan usia juga tidak terlalu berkorelasi.

Menganalisa tingkat resiko per fitur (Usia, Tekanan darah, Kadar gula darah, Suhu tubuh, dan Detak jantung)
Menampilkan grafik karakteristik terhadap tingkat resiko:

Mencari korelasi antara usia, tekanan darah, kadar gula darah dan detak jantung
    correlation = data[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'HeartRate', 'RiskLevel']].corr(numeric_only=True)
>> Memetakan korelasi dalam bentuk heatmap. 
>> Memetakan korelasi dalam bentuk scatterplot. \


## **Data Preparation**
### Train-Test-Split
Membagi dataset menjadi data latih (train) dan data uji (test) dengan perbandingan 80:20. Ratio ini dianggap cukup karena besar data yang adalah 1014.

### Standarisasi
Melakukan standarisasi pada fitur numerik yaitu 'Age', 'SystolicBP', 'DiastolicBP', 'BS', 'HeartRate' dengan menggunakan teknik StandarScaler dari library Scikitlearn. Standarisasi membantu membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma sehingga hasil model akan lebih baik.


## **Model Development**
Tahap selanjutnya: mengembangkan model machine learning dengan empat algoritma. Kemudian, kita akan mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Keempat algoritma yang akan kita gunakan, antara lain:
1. Multinominal logistic regression.
2. K-Nearest Neighbor.
3. Random Forest.
4. Decision Tree Classifier.

### Hyperparameter tuning 
Menggunakan XGBClassifier


## **Model Evaluation**
Dengan evaluasi XGBClassifier dan juga perbandingan 4 model dengan metrik Mean Squared Error, didapatkan bahwa model Random Forest memberikan angka error paling kecil di antara 4 model lainnya, sedangkan model dari Logistic Regression memberikan angka error paling besar. 
Karena itu, model Random Forest yang akan dipilih sebagai model terbaik dalam memetakan tingkat resiko kehamilan. 

### Menguji model 
Dengan prediksi, didapatkan bahwa hasil dari model Random Forest tidak jauh berbeda dengan nilai asli. Walaupun Logistic Regression dan Decision Tree memberikan hasil akurat 0 seperti pada nilai asli, namun tidak menjamin hasil percobaan selanjutnya karena saat evaluasi keduanya memberikan nilai error lebih besar daripada Random Forest. 
