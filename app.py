import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    layout="wide",
    page_title="EduMate: Sistem Rekomendasi Pembelajaran",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_all_artifacts():
    """Memuat model TensorFlow yang sudah dilatih, encoder, scaler, dan data konten."""
    try:
        model_path = 'model_artifacts/recs_model.h5'
        le_path = 'model_artifacts/label_encoders.pkl'
        scaler_path = 'model_artifacts/scalers.pkl'
        features_path = 'model_artifacts/features_info.pkl'
        unique_cat_path = 'model_artifacts/unique_categories.pkl'
        data_konten_path = 'model_artifacts/data_konten_for_recs.pkl'

        # Memuat model TensorFlow
        model = load_model(model_path)

        # Memuat LabelEncoders
        with open(le_path, 'rb') as f:
            label_encoders = pickle.load(f)

        # Memuat StandardScaler
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)

        # Memuat informasi fitur (nama-nama kolom yang digunakan model)
        with open(features_path, 'rb') as f:
            features_info = pickle.load(f)

        # Memuat kategori unik (untuk dropdown di UI)
        with open(unique_cat_path, 'rb') as f:
            unique_categories = pickle.load(f)

        # Memuat data_konten yang sudah disiapkan untuk rekomendasi
        data_konten_for_recs = pd.read_pickle(data_konten_path)
        
        if 'kesulitan' in data_konten_for_recs.columns:
            data_konten_for_recs['tingkat_kesulitan'] = data_konten_for_recs['kesulitan']
        else:
            st.warning("Kolom 'kesulitan' tidak ditemukan di data_konten_for_recs. Pastikan nama kolom benar.")
            data_konten_for_recs['tingkat_kesulitan'] = 'Unknown' # Fallback

        return model, label_encoders, scalers, features_info, unique_categories, data_konten_for_recs
    except Exception as e:
        st.error(f"❌ Error saat memuat artefak model atau data: {e}. Pastikan folder 'model_artifacts' dan isinya ada dan benar.")
        st.stop() # Menghentikan eksekusi aplikasi Streamlit jika gagal memuat artefak.

# Memuat semua artefak saat aplikasi dimulai
model, label_encoders, scalers, features_info, unique_categories, data_konten_for_recs = load_all_artifacts()

# Ekstrak nama fitur dari features_info untuk kemudahan akses
user_categorical_tf_features = features_info.get('user_categorical_tf_features', [])
user_numerical_tf_features = features_info.get('user_numerical_tf_features', [])
interaction_numerical_tf_features = features_info.get('interaction_numerical_tf_features', [])
content_categorical_tf_features = features_info.get('content_categorical_tf_features', [])
content_numerical_tf_features = features_info.get('content_numerical_tf_features', [])

# Menggabungkan fitur numerik pengguna dan interaksi untuk input model
combined_user_numerical_tf_features_names = sorted(list(set(user_numerical_tf_features + interaction_numerical_tf_features)))

# --- Inisialisasi Session State untuk Histori Rekomendasi ---
if 'recommendation_history' not in st.session_state:
    st.session_state.recommendation_history = []
if 'total_recommendations_served' not in st.session_state:
    st.session_state.total_recommendations_served = 0
if 'total_completed_recommendations' not in st.session_state:
    st.session_state.total_completed_recommendations = 0
if 'current_recommendations_display' not in st.session_state:
    st.session_state.current_recommendations_display = pd.DataFrame()


# --- Fungsi Callback untuk Menandai Selesai ---
def mark_as_completed(judul_konten, predicted_score, original_rating):
    st.session_state.recommendation_history.append({
        'judul': judul_konten,
        'tanggal': pd.Timestamp.now().strftime('%d %B %Y'),
        'status': 'Selesai',
        'rating_konten_asli': original_rating
    })
    st.session_state.total_completed_recommendations += 1
    st.success(f"Konten '{judul_konten}' ditandai selesai dan ditambahkan ke histori!")


# --- Fungsi Preprocessing untuk Rekomendasi (Batch Prediction) ---
def preprocess_for_recommendation(user_preferences, all_content_df, model_input_names):
   
    # 1. Simpan data konten asli untuk tampilan (UNSCALED)
    display_cols_for_original = ['id_konten', 'judul', 'durasi', 'mata_kuliah', 'platform', 'format', 'kesulitan', 'rating_pengguna']
    original_content_for_display = all_content_df[all_content_df.columns.intersection(display_cols_for_original)].copy()

    # Tambahkan tingkat_kesulitan ke original_content_for_display jika belum ada
    if 'kesulitan' in original_content_for_display.columns and 'tingkat_kesulitan' not in original_content_for_display.columns:
        original_content_for_display['tingkat_kesulitan'] = original_content_for_display['kesulitan']
    elif 'tingkat_kesulitan' not in original_content_for_display.columns:
        original_content_for_display['tingkat_kesulitan'] = 'Unknown'

    # Filter all_content_df berdasarkan mata_kuliah yang dipilih user
    selected_mata_kuliah = user_preferences.get('mata_kuliah')
    if selected_mata_kuliah and selected_mata_kuliah != 'Lainnya':
        filtered_content_df = all_content_df[all_content_df['mata_kuliah'] == selected_mata_kuliah].copy()
        if filtered_content_df.empty:
            st.warning(f"Tidak ada konten ditemukan untuk Mata Kuliah: '{selected_mata_kuliah}'. Mencari di semua mata kuliah.")
            filtered_content_df = all_content_df.copy() 
    else:
        filtered_content_df = all_content_df.copy() 

    # Siapkan DataFrame untuk PREDIKSI (melakukan merge dengan user_pref)
    user_pref_df = pd.DataFrame([user_preferences])
    # Gunakan filtered_content_df di sini
    df_for_prediction_input = pd.merge(user_pref_df.assign(key=1), filtered_content_df.assign(key=1), on='key', suffixes=('_user', '_content')).drop('key', axis=1)
    
    # --- Pra-pemrosesan: Label Encoding pada df_for_prediction_input ---
    for feature_name, le in label_encoders.items():
        if feature_name in df_for_prediction_input.columns:
            df_for_prediction_input[f'{feature_name}_encoded'] = df_for_prediction_input[feature_name].apply(
                lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else (
                    le.transform(['Unknown'])[0] if 'Unknown' in le.classes_ else len(le.classes_) - 1
                )
            )
        elif f'{feature_name}_user' in df_for_prediction_input.columns:
            df_for_prediction_input[f'{feature_name}_encoded'] = df_for_prediction_input[f'{feature_name}_user'].apply(
                lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else (
                    le.transform(['Unknown'])[0] if 'Unknown' in le.classes_ else len(le.classes_) - 1
                )
            )
            df_for_prediction_input = df_for_prediction_input.drop(columns=[f'{feature_name}_user'])
        else:
            st.warning(f"⚠️ Fitur '{feature_name}' (untuk encoding) tidak ditemukan di DataFrame prediksi. Menetapkan encoded_value default.")
            default_category = unique_categories.get(feature_name, ['Unknown'])[0]
            default_encoded_value = le.transform([str(default_category)])[0] if str(default_category) in le.classes_ else (
                le.transform(['Unknown'])[0] if 'Unknown' in le.classes_ else 0
            )
            df_for_prediction_input[f'{feature_name}_encoded'] = default_encoded_value

    # --- Pra-pemrosesan: Scaling Fitur Numerik pada df_for_prediction_input ---
    all_numerical_features_in_df = [
        f for f in combined_user_numerical_tf_features_names + content_numerical_tf_features
        if f in df_for_prediction_input.columns
    ]

    if 'all_numerical' in scalers and all_numerical_features_in_df:
        df_numerical_subset = df_for_prediction_input[all_numerical_features_in_df].copy()
        df_numerical_subset = df_numerical_subset.reindex(columns=scalers['all_numerical'].feature_names_in_, fill_value=0.0)
        scaled_values = scalers['all_numerical'].transform(df_numerical_subset.values)
        df_for_prediction_input[all_numerical_features_in_df] = scaled_values
    else:
        st.warning("⚠️ Tidak ada scaler numerik atau fitur numerik yang ditemukan untuk scaling. Melanjutkan tanpa scaling numerik.")

    # --- Menyiapkan Input dalam Format yang Diharapkan oleh Model TensorFlow dari df_for_prediction_input ---
    ordered_inputs = []
    try:
        for name in model_input_names:
            if name.startswith('user_') and name.endswith('_encoded'):
                feature_name_encoded = name.replace('user_', '')
                if feature_name_encoded in df_for_prediction_input.columns:
                    ordered_inputs.append(df_for_prediction_input[feature_name_encoded].values.reshape(-1, 1))
                else:
                    raise ValueError(f"Kolom kategorikal pengguna tidak ditemukan di df_for_prediction_input: {feature_name_encoded}")
            elif name == 'user_combined_numerical_features':
                if combined_user_numerical_tf_features_names and all(f in df_for_prediction_input.columns for f in combined_user_numerical_tf_features_names):
                    ordered_inputs.append(df_for_prediction_input[combined_user_numerical_tf_features_names].values)
                else:
                    raise ValueError(f"Kolom numerik gabungan pengguna tidak ditemukan atau tidak lengkap di df_for_prediction_input: {combined_user_numerical_tf_features_names}")
            elif name.startswith('content_') and name.endswith('_encoded'):
                feature_name_encoded = name.replace('content_', '')
                if feature_name_encoded in df_for_prediction_input.columns:
                    ordered_inputs.append(df_for_prediction_input[feature_name_encoded].values.reshape(-1, 1))
                else:
                    raise ValueError(f"Kolom kategorikal konten tidak ditemukan di df_for_prediction_input: {feature_name_encoded}")
            elif name == 'content_numerical_features':
                if content_numerical_tf_features and all(f in df_for_prediction_input.columns for f in content_numerical_tf_features):
                    ordered_inputs.append(df_for_prediction_input[content_numerical_tf_features].values)
                else:
                    raise ValueError(f"Kolom numerik konten tidak ditemukan atau tidak lengkap di df_for_prediction_input: {content_numerical_tf_features}")
            else:
                raise ValueError(f"Nama input layer model tidak dikenal atau tidak ditangani: {name}")

    except ValueError as ve:
        st.error(f"❌ Error dalam menyiapkan input model: {ve}")
        return None, None
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan tak terduga saat menyiapkan input model: {e}")
        return None, None

    # Menggabungkan kembali dengan data asli untuk tampilan
    # Pastikan id_konten ada di df_for_prediction_input untuk merge
    df_for_prediction_input_with_id = df_for_prediction_input[['id_konten']].copy()
    final_df_info = df_for_prediction_input_with_id.merge(original_content_for_display, on='id_konten', how='left')
    
    return ordered_inputs, final_df_info

# --- Mendapatkan Urutan Nama Input Layer dari Model ---
model_input_names = [inp.name.split(':')[0] for inp in model.inputs]


# --- Streamlit UI dan Logic Aplikasi ---

st.title("💡 EduMate: Sistem Rekomendasi Belajar Cerdas")
st.write("Temukan referensi belajar terbaik yang sesuai dengan preferensi Anda.")

# --- Sidebar untuk Input Preferensi Belajar ---
st.sidebar.header("Masukan Preferensi Belajar Anda")
user_input_dict = {}

# Input preferensi belajar yang disederhanakan
user_input_dict['jurusan'] = st.sidebar.selectbox(
    "Jurusan",
    unique_categories.get('jurusan', ['Umum'])
)
user_input_dict['mata_kuliah'] = st.sidebar.selectbox(
    "Mata Kuliah yang Ingin Dikuasai",
    unique_categories.get('mata_kuliah', ['Lainnya'])
)
user_input_dict['waktu_belajar_per_hari'] = st.sidebar.number_input(
    "Rata-rata Waktu Belajar per Hari (Jam)",
    min_value=0.0, max_value=24.0, value=2.0, step=0.5
)

# --- Input user lainnya yang diperlukan model, tapi tidak diinput langsung di UI ---
user_input_dict['ipk_terakhir'] = 3.0
user_input_dict['device_preference'] = unique_categories.get('device_preference', ['Desktop'])[0]
user_input_dict['learning_style'] = unique_categories.get('learning_style', ['Visual'])[0]
user_input_dict['goal'] = unique_categories.get('goal', ['Meningkatkan Skill'])[0]
user_input_dict['ketersediaan_belajar'] = unique_categories.get('ketersediaan_belajar', ['Pagi Hari'])[0]

user_input_dict['watch_ratio'] = 1.0


# --- Tabs untuk Fitur Utama: Rekomendasi dan Pencarian ---
st.markdown("---")
tab1, tab2 = st.tabs(["🏡 Beranda & Rekomendasi", "🔍 Pencarian Lanjutan"])

with tab1:
    st.header("Rekomendasi Terbaik untuk Anda")
    st.write("Dapatkan daftar konten pembelajaran yang paling sesuai dengan preferensi Anda.")

    with st.form(key='recommendation_form'):
        submit_recommendation = st.form_submit_button("Dapatkan Rekomendasi", use_container_width=True)

        if submit_recommendation:
            st.info("💡 Memproses preferensi Anda dan mencari rekomendasi terbaik... Mohon tunggu sebentar.")

            inputs_for_prediction, df_with_content_info = preprocess_for_recommendation(user_input_dict, data_konten_for_recs, model_input_names)

            if inputs_for_prediction is not None and df_with_content_info is not None:
                try:
                    predicted_scores = model.predict(inputs_for_prediction).flatten()
                    df_with_content_info['predicted_feedback_score'] = predicted_scores 
                    top_recommendations = df_with_content_info.sort_values(by='predicted_feedback_score', ascending=False).head(5)
                    
                    st.session_state.current_recommendations_display = top_recommendations
                    st.session_state.total_recommendations_served += len(top_recommendations)

                    st.rerun() 

                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat memprediksi rekomendasi: {e}")
            else:
                st.warning("Gagal memproses input preferensi Anda. Periksa pesan kesalahan di atas.")
    
    # --- Tampilan Rekomendasi (di luar form) ---
    if not st.session_state.current_recommendations_display.empty:
        st.subheader("Berikut 5 rekomendasi EduMate terbaik yang sesuai untuk Anda:")
        for index, row in st.session_state.current_recommendations_display.iterrows():
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                st.markdown(f"#### {row['judul']}")
                st.write(f"**Mata Kuliah:** {row.get('mata_kuliah', 'N/A')}")
                st.write(f"**Platform:** {row.get('platform', 'N/A')} | **Format:** {row.get('format', 'N/A')}")
                st.write(f"**Durasi:** {row.get('durasi', 'N/A')} menit | **Tingkat Kesulitan:** {row.get('tingkat_kesulitan', 'N/A')}") 
            with col2:
                st.metric(label="Rating Pengguna", value=f"{row.get('rating_pengguna', 0.0):.1f} / 5")
                
                # Tombol "Tandai Selesai"
                st.button(
                    "Tandai Selesai", 
                    key=f"complete_rec_{row['id_konten']}",
                    on_click=mark_as_completed,
                    args=(row['judul'], row['predicted_feedback_score'], row.get('rating_pengguna', 0.0))
                )
            st.markdown("---")
    else:
        if st.session_state.total_recommendations_served == 0:
            st.info("Silakan masukkan preferensi Anda di sidebar dan klik 'Dapatkan Rekomendasi'.")

    st.markdown("---")
    st.markdown("### 📊 Dashboard Pembelajaran")
    st.write("Ringkasan aktivitas belajar Anda.")

    total_recs = st.session_state.total_recommendations_served
    total_completed = st.session_state.total_completed_recommendations

    completion_percentage = (total_completed / total_recs) * 100 if total_recs > 0 else 0

    st.subheader("Status Belajar Anda:")
    st.info(f"Anda telah menandai **{total_completed}** dari **{total_recs}** rekomendasi sebagai selesai. ({completion_percentage:.1f}%)")
    st.progress(completion_percentage / 100)

    st.subheader("Histori Materi yang Telah Diakses:")
    if st.session_state.recommendation_history:
        history_df = pd.DataFrame(st.session_state.recommendation_history)
        history_df = history_df[['judul', 'tanggal', 'status', 'rating_konten_asli']].rename(columns={'rating_konten_asli': 'Rating Konten Asli'})
        st.dataframe(history_df.tail(5).sort_values(by='tanggal', ascending=False).reset_index(drop=True), use_container_width=True)
        if len(st.session_state.recommendation_history) > 5:
            st.info(f"Menampilkan 5 entri terbaru. Total ada {len(st.session_state.recommendation_history)} entri dalam histori.")
    else:
        st.info("Anda belum menandai konten rekomendasi apapun sebagai selesai.")


with tab2:
    st.header("Pencarian Lanjutan")
    st.write("Cari sumber belajar secara manual dengan filter kategori dan tingkat kesulitan.")

    search_query = st.text_input("Cari berdasarkan Judul Konten", placeholder="Contoh: Python, Deep Learning, Statistik...")

    st.subheader("Filter Kategori")
    selected_mata_kuliah = st.multiselect(
        "Mata Kuliah",
        unique_categories.get('mata_kuliah', [])
    )
    selected_platform = st.multiselect(
        "Platform",
        unique_categories.get('platform', [])
    )
    selected_format = st.multiselect(
        "Format Konten",
        unique_categories.get('format', [])
    )

    st.subheader("Filter Tingkat Kesulitan")
    selected_kesulitan = st.multiselect(
        "Tingkat Kesulitan",
        unique_categories.get('kesulitan', [])
    )

    if st.button("Terapkan Filter", use_container_width=True):
        filtered_content = data_konten_for_recs.copy()

        if search_query:
            filtered_content = filtered_content[filtered_content['judul'].str.contains(search_query, case=False, na=False)]

        if selected_mata_kuliah:
            filtered_content = filtered_content[filtered_content['mata_kuliah'].isin(selected_mata_kuliah)]
            
        if selected_platform:
            filtered_content = filtered_content[filtered_content['platform'].isin(selected_platform)]
            
        if selected_format:
            filtered_content = filtered_content[filtered_content['format'].isin(selected_format)]
            
        if selected_kesulitan:
            filtered_content = filtered_content[filtered_content['kesulitan'].isin(selected_kesulitan)]



        st.subheader(f"Hasil Pencarian ({len(filtered_content)} konten ditemukan):")
        if not filtered_content.empty:
            for index, row in filtered_content.head(10).iterrows():
                st.markdown(f"#### {row['judul']}")
                st.write(f"**Mata Kuliah:** {row.get('mata_kuliah', 'N/A')}")
                st.write(f"**Platform:** {row.get('platform', 'N/A')} | **Format:** {row.get('format', 'N/A')}")
                st.write(f"**Durasi:** {row.get('durasi', 'N/A')} menit | **Tingkat Kesulitan:** {row.get('tingkat_kesulitan', 'N/A')}")
                st.write(f"**Rating Pengguna:** {row.get('rating_pengguna', 0.0):.1f} / 5")
                st.markdown("---")
            if len(filtered_content) > 10:
                st.info(f"Menampilkan 10 dari {len(filtered_content)} hasil. Sesuaikan filter untuk melihat lebih banyak.")
        else:
            st.warning("Tidak ada konten yang sesuai dengan filter Anda. Coba sesuaikan kriteria pencarian atau filter.")

st.markdown("---")
st.markdown("Pengembang: EduMate Team")
