from PyQt5.QtWidgets import *
from grafik_sonuc_fft import Ui_Form
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

class GrafikShowFFT(QWidget):
    def __init__(self, kanal_adi, kanal_verisi, freqs_filt):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        
        # Matplotlib grafiği için bir canvas oluştur
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        self.toolbar = NavigationToolbar(self.canvas, self)

        # Arayüze canvas'ı ekle
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Grafiği çiz
        self.grafik_ciz(kanal_adi, kanal_verisi, freqs_filt)

    def grafik_ciz(self, kanal_adi, kanal_verisi, freqs_filt):
        self.ax.clear()
        self.ax.plot(freqs_filt, kanal_verisi, label=kanal_adi)
        self.ax.set_title(f"{kanal_adi} Kanalı FFT Verisi")
        self.ax.set_xlabel("Zaman")
        self.ax.set_ylabel("Sinyal Frekansı")
        self.ax.legend()
        self.canvas.draw()