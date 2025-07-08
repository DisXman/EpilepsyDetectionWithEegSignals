import sys
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from anamenu import Ui_MainWindow 
from tahmin_tik_py import TahminWindow
from grafik_tik_py import GrafikSecme




import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
import mne
import tensorflow as tf
from tensorflow import keras







class AnaPencere(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.tahmin1_ac_form = TahminWindow()
        self.grafik_ciz_form = GrafikSecme()
        

        self.ui.tahminet_btn.clicked.connect(self.tahmin1_git)
        self.ui.grafikciz_btn.clicked.connect(self.grafik_ciz_goster)


    def grafik_ciz_goster(self):
        self.grafik_ciz_form.show()

    def tahmin1_git(self):
        self.tahmin1_ac_form.show()


if __name__ == "__main__":
    app = QApplication([])
    pencere = AnaPencere()
    pencere.show()
    sys.exit(app.exec_())
