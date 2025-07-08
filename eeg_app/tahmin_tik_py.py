# from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import mne
import tensorflow as tf
from tensorflow import keras
import joblib


from PyQt5.QtWidgets import *
from tahmin_tik import Ui_MainWindow
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sonuc_py import SonucShow



class TahminWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tahmintik_form = Ui_MainWindow()
        self.tahmintik_form.setupUi(self)
        self.tahmintik_form.btn_csv_sec.clicked.connect(self.csv_dosyasi_sec)
        
        self.sonuc_form = SonucShow()
                
    
           
    
    
    def csv_dosyasi_sec(self):
        dosya_yolu, _ = QFileDialog.getOpenFileName(self, "CSV Dosyası Seç", "", "CSV Dosyaları (*.csv)")
        if dosya_yolu:
            self.tahmintik_form.label_path.setText(str(dosya_yolu))
        if dosya_yolu:
            def bandpass_filter(data, lowcut=1, highcut=40, fs=128, order=4):
                nyquist = 0.5 * fs
                low = lowcut / nyquist
                high = highcut / nyquist
                b, a = butter(order, [low, high], btype='band')
                return filtfilt(b, a, data)

            def filtrele_tek_dosya(girdi_yolu, fs=128):
                df = pd.read_csv(girdi_yolu)
                columns_list = ["AF3","AF4","F3","F4","F7","F8","FC5","FC6","O1","O2","P7","P8","T7","T8"]
                df = df[columns_list]

                for col in df.columns:
                    if col != "label":
                        df[col] = bandpass_filter(df[col], fs=fs)
                return df

            # Örnek kullanım:
            # girdi_csv = "/kaggle/input/control12/signal-5.csv"  # giriş dosyası

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

            

            # Model ve scaler'ı yükle
            model = keras.models.load_model("lstm_epilepsi_modeli.keras")
            scaler = joblib.load("scaler.pkl")

            # Dışarıdan yeni veri oku (örnek)
            # data = pd.read_csv("/kaggle/input/epilepsy/epilepsy_ica_applied_data-20250606T212407Z-1-001/epilepsy_ica_applied_data/epilepsy-10.csv")  # kendi dosya yoluna göre ayarla

            # Kullanacağımız EEG kanalları (model eğitimi ile aynı)
            columns_list = ["AF3","AF4","F3","F4","F7","F8","FC5","FC6","O1","O2","P7","P8","T7","T8"]
            df = ica_data[columns_list]
            

            # Modelde kullandığımız pencere boyutu ve kayma ile aynı olmalı
            pencere_boyutu = 128
            kayma = 64

            # Veriyi pencereye böl (aynı şekilde)
            def pencereye_bol_yeni(veri, pencere_boyutu=128, kayma=64):
                X = []
                veri = veri.values
                for i in range(0, len(veri) - pencere_boyutu + 1, kayma):
                    pencere = veri[i:i+pencere_boyutu]
                    X.append(pencere)
                return np.array(X)

            X_yeni = pencereye_bol_yeni(df, pencere_boyutu, kayma)

            # Ölçeklendirme için reshape
            n_ornek, n_adim, n_kanal = X_yeni.shape
            X_yeni_reshaped = X_yeni.reshape(-1, n_kanal)

            # Ölçeklendir (scaler fit edilmemeli, sadece transform edilmeli)
            X_yeni_scaled = scaler.transform(X_yeni_reshaped)

            # Yeniden şekillendir
            X_yeni_scaled = X_yeni_scaled.reshape(n_ornek, n_adim, n_kanal)

            # Tahmin et
            tahminler = model.predict(X_yeni_scaled)
            # print(tahminler)
            # Tahminleri 0/1 olarak yuvarla (binary classification için)
            tahmin_etiketleri = (tahminler > 0.5).astype(int)
            tahmin_etiketleri = np.array(tahmin_etiketleri)
            bir = 0
            sifir = 0
            toplam = len(tahmin_etiketleri)
            for i in tahmin_etiketleri:
                if i == 1:
                    bir = bir +1
                elif i == 0 :
                    sifir = sifir + 1

            if sifir <= bir :
                self.sonuc = float(bir / toplam) 
                print(f"{self.sonuc} oranında  epilepsi")
                self.sonuc_form.set_sonuc(self.sonuc,"epilepsi")
                self.sonuc_form.show()
                
            else:
                self.sonuc = float(sifir / toplam) 
                print(f"{self.sonuc} oranında  normal")
                self.sonuc_form.set_sonuc(self.sonuc, "normal")
                self.sonuc_form.show()
            
        
                             
    # def sonuc_show(self):
    #     self.sonuc_form.show()      
            
            
            