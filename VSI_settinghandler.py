from PyQt5.QtCore import pyqtSignal,Qt
from PyQt5.QtWidgets import QWidget,QSizePolicy,QVBoxLayout,QGroupBox

class SettingHandler(QWidget):
    
    command = pyqtSignal(str,list)
    
    def __init__(self,combi_setting,parent=None):
        super(self.__class__, self).__init__(parent)
         
        self.setting_class = combi_setting
        self.initialize()
        
    def initialize(self):    
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.vbox = QVBoxLayout()
        self.vbox.setAlignment(Qt.AlignTop)
        self.setLayout(self.vbox) 
        
    def addSetting(self,name,specs):
        if specs is not None:
            setting = self.setting_class(name,specs)
            setting.command.connect(self.command.emit)
            self.vbox.addWidget(setting)
    
    def removeSetting(self,index):        
        item = self.vbox.takeAt(index)
        if item is not None:
            widget = item.widget()
            widget.hide()
            widget.deleteLater()
        
    def quitSettings(self):
        for box in self.findChildren(QGroupBox):
            box.hide()
            box.deleteLater()
