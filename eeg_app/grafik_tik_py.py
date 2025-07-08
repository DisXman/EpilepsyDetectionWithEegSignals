from PyQt5.QtWidgets import *
from grafik_tik import Ui_MainWindow
import pandas as pd
from grafik_sonuc_py import GrafikShow
from grafik_sonuc_fft_py import GrafikShowFFT


import numpy as np
from scipy.signal import butter, filtfilt
import mne
import tensorflow as tf
from tensorflow import keras
import joblib

from scipy.fft import fft, fftfreq


class GrafikSecme(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.df = None
        self.ui.btn_csv_sec.clicked.connect(self.csv_dosya_sec)
        self.ui.btn_grafik_goster.clicked.connect(self.grafik_goster)
        self.ui.stft_buton.clicked.connect(self.stft_grafik_goster)
        
    def csv_dosya_sec(self):
        dosya_yolu,_ = QFileDialog.getOpenFileName(self, "CSV Dosyanı Seç", "", "CSV Dosyaları(*.csv)")
        if dosya_yolu:
            # self.df = pd.read_csv(dosya_yolu)
            self.ui.label_path.setText(str(dosya_yolu))
        if dosya_yolu:
            def bandpass_filter(data, lowcut=1, highcut=40, fs=128, order=4):
                nyquist = 0.5 * fs
                low = lowcut / nyquist
                high = highcut / nyquist
                b, a = butter(order, [low, high], btype='band')
                return filtfilt(b, a, data)

            def filtrele_tek_dosya(dosya_yolu, fs=128):
                df = pd.read_csv(dosya_yolu)
                columns_list = ["AF3","AF4","F3","F4","F7","F8","FC5","FC6","O1","O2","P7","P8","T7","T8"]
                df = df[columns_list]

                for col in df.columns:
                    if col != "label":
                        df[col] = bandpass_filter(df[col], fs=fs)
                return df

        

            bandpass_filtered_data = filtrele_tek_dosya(dosya_yolu)



            

            def convert_csv_to_mne(df, sfreq=128):
                eeg_channels = [col for col in df.columns if col not in ['label']]
                data = df[eeg_channels].to_numpy().T  # shape: (n_channels, n_samples)
                info = mne.create_info(ch_names=eeg_channels, sfreq=sfreq, ch_types='eeg')
                raw = mne.io.RawArray(data, info)

                montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage)

                return raw

            def apply_ica(raw):
                ica = mne.preprocessing.ICA(n_components=0.99, random_state=97, max_iter=800)
                ica.fit(raw)
                ica.exclude = [0, 3, 4, 6, 8, 10, 11]  # örnek olarak, gözle seçilmesi önerilir
                cleaned = ica.apply(raw.copy())
                return cleaned

            # 📌 Yalnızca bir dosya üzerinde ICA uygula
            def apply_ica_to_single_file(df, sfreq=128):
                # df = pd.read_csv(girdi_yolu)

                raw = convert_csv_to_mne(df, sfreq=sfreq)
                cleaned = apply_ica(raw)

                data = cleaned.get_data()
                channels_name = cleaned.ch_names
                df_cleaned = pd.DataFrame(data.T, columns=channels_name)

                # Etiket kolonu varsa tekrar ekleyebilirsin
                # if 'label' in df.columns:
                #     df_cleaned['label'] = df['label'].iloc[0]

                return df_cleaned



            ica_data = apply_ica_to_single_file(bandpass_filtered_data)

            

        

            # Dışarıdan yeni veri oku (örnek)
            # data = pd.read_csv("/kaggle/input/epilepsy/epilepsy_ica_applied_data-20250606T212407Z-1-001/epilepsy_ica_applied_data/epilepsy-10.csv")  # kendi dosya yoluna göre ayarla

            # Kullanacağımız EEG kanalları (model eğitimi ile aynı)
            columns_list = ["AF3","AF4","F3","F4","F7","F8","FC5","FC6","O1","O2","P7","P8","T7","T8"]
            self.df = ica_data[columns_list]
            self.ui.comboBox_kanal.clear()
            self.ui.comboBox_kanal.addItems(self.df.columns)
            
            
    def grafik_goster(self):
        # dosya_yolu = QFileDialog.getOpenFileName(self, "CSV Dosyanı Seç", "", "CSV Dosyaları(*.csv)")
        # if dosya_yolu:
        if self.df is not None:
            kanal_adi = self.ui.comboBox_kanal.currentText()
            veriler = self.df[kanal_adi].values
            self.grafik_goster_form = GrafikShow(kanal_adi, veriler)
            self.grafik_goster_form.show()
            
    def stft_grafik_goster(self):
        if self.df is not None:
            kanal_adi = self.ui.comboBox_kanal.currentText()
            veriler = self.df[kanal_adi].values
            def compute_fft(veriler, fs):
                N = len(veriler)
                freqs = fftfreq(N, 1 / fs)[:N // 2]
                fft_values = np.abs(fft(veriler))[:N // 2]
                return freqs, fft_values
            freqs_filt, fft_filt = compute_fft(veriler, 128)
            
            self.grafik_goster_form = GrafikShowFFT(kanal_adi, fft_filt, freqs_filt)
            self.grafik_goster_form.show()
        
            
            
            
            
    
