# 💡 EduMate: Sistem Rekomendasi Belajar Cerdas
Selamat datang di repositori EduMate, sebuah aplikasi web berbasis Streamlit yang dirancang untuk membantu Anda menemukan konten pembelajaran paling relevan sesuai dengan preferensi belajar pribadi Anda.

## ✨ Fitur Utama
- Rekomendasi Personalisasi: Dapatkan saran konten pembelajaran yang disesuaikan berdasarkan jurusan, mata kuliah yang ingin dikuasai, waktu belajar, gaya belajar, dan preferensi perangkat.

- Pencarian Lanjutan: Cari konten secara manual dengan filter berdasarkan mata kuliah, platform, format konten, dan tingkat kesulitan.

- Histori Pembelajaran: Lacak konten yang telah Anda tandai selesai dan lihat ringkasan kemajuan belajar Anda.

- Antarmuka Pengguna Interaktif: Dibangun dengan Streamlit untuk pengalaman yang intuitif dan responsif.

## ⚙️ Cara Kerja Sistem Rekomendasi
Sistem EduMate menggunakan model rekomendasi hybrid yang dilatih dengan TensorFlow/Keras. Model ini mempertimbangkan berbagai fitur pengguna dan konten untuk memprediksi seberapa relevan dan bermanfaat suatu konten bagi seorang individu. Prediksi ini kemudian digunakan untuk mengurutkan dan menyajikan rekomendasi terbaik.

## Setup Environment - Shell/Terminal
Sebelum menginstall library, disarankan untuk menggunakan virtual environment agar dependensi tetap terisolasi.
Jalankan perintah berikut :
python -m venv venv
venv\Scripts\activate

## Setup Library
Setelah mengaktifkan virtual environment, install semua dependensi dari file requirements.txt dengan perintah berikut :
pip install -r requirements.txt

## Jalankan Aplikasi
Setelah semua dependensi terinstal dan artefak model tersedia, Anda dapat menjalankan aplikasi Streamlit dari direktori utama proyek:
streamlit run app.py

## 📁 Struktur Proyek
EduMate-App/
|-- app.py
|-- model_artifacts/
|   |-- recs_model.h5
|   |-- label_encoders.pkl
|   |-- scalers.pkl
|   |-- features_info.pkl
|   |-- unique_categories.pkl
|   `-- data_konten_for_recs.pkl
|-- requirements.txt
|-- README.md
`-- edumate.ipynb
