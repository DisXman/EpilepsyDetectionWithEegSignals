from PyQt5.QtWidgets import *
from sonuc import Ui_Form


class SonucShow(QWidget):
    def __init__(self):
        super().__init__()
        self.sonuc_form = Ui_Form()
        self.sonuc_form.setupUi(self)
    
    def set_sonuc(self, sonuc_orani, text):
        self.sonuc_form.sonuc_label.setText(f"{sonuc_orani:.2%} olasılıkla {text}")