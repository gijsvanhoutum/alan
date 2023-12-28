from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import os
# ICONS at: https://specifications.freedesktop.org/icon-naming-spec/icon-naming-spec-latest.html#names

class Gui(QMainWindow):    
    
    generator = pyqtSignal(str)
    transform = pyqtSignal(str)
    command = pyqtSignal(str)
    
    def __init__(self,view,setting,device,transform,parent=None):
        super(self.__class__, self).__init__(parent)

        self.setting = setting
        self.view = view

        self._set_constants()
        self._create_dock_widgets()
        self._create_menus(device,transform)
        
        self.initial()
        
        self.show()
        
    def _set_constants(self):
        self.icon_path = "../icons/"

        self.setting_title = "Setting"
        self.view_title = "View"
        self.source_title = "Select Source"
        self.source_device = "Device"
        self.source_file = "File"
        self.algo_title = "Add Algorithm"
        self.sink_title = "Select Sink"

    def initial(self):
        self.settings_dock.hide()        
        self.views_dock.hide()
        self.AlgoMenu.menuAction().setVisible(False)
        self.SinkMenu.menuAction().setVisible(False)
        
    def sizeHint(self):
        return QSize(640,480)
      
    def _create_dock_widgets(self):
        scrol = VerticalScrollArea(self.setting)        
        self.settings_dock = DockWidget(scrol)
        self.settings_dock.setDefaultTitle(self.setting_title)

        self.views_dock = DockWidget(self.view)
        self.views_dock.setDefaultTitle(self.view_title)

        self.addDockWidget(Qt.LeftDockWidgetArea,self.settings_dock) 
        self.addDockWidget(Qt.RightDockWidgetArea,self.views_dock)            
        self.setDockNestingEnabled(True)

    def _create_menus(self,source,transform):
        self.SourceMenu = self.menuBar().addMenu("&"+self.source_title)
        self.AlgoMenu = self.menuBar().addMenu("&"+self.algo_title)
        self.SinkMenu = self.menuBar().addMenu("&"+self.sink_title)
        
        for src_cls in source.getClassList():
            name = src_cls.getClassName()
            icon = src_cls.getClassIcon()
            action = self.createAction(name,icon)                 
            self.SourceMenu.addAction(action)
            l = lambda state,n=name: self._open(n)
            action.triggered.connect(l)   

        for trf_cls in transform.getClassList():
            name = trf_cls.getClassName()
            action = self.createAction(name) 
            self.AlgoMenu.addAction(action)            
            l = lambda state,n=name: self.transform.emit(n)
            action.triggered.connect(l)                     

    def _open(self,name):      
        self.views_dock.show()
        self.settings_dock.show()    
        self.AlgoMenu.menuAction().setVisible(True)
        self.generator.emit(name)
            
    def createAction(self, text, icon=None,tip=None,shortcut=None):                         
        action = QAction(text, self)
        if icon is not None:
            for f_name in os.listdir(self.icon_path):
                if f_name.startswith(icon): 
                    ic = QIcon(self.icon_path+f_name)
                    action.setIcon(ic)
            
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        else:
            action.setToolTip(text)
            action.setStatusTip(text)

        return action

class VerticalScrollArea(QScrollArea):
    def __init__(self,widget):
        super(self.__class__, self).__init__()   
        self.setFrameShadow(QFrame.Plain)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.contents = widget
        self.setWidget(self.contents)
        self.contents.installEventFilter(self)

    def eventFilter(self,o,e):
        if o == self.contents and e.type() == QEvent.Resize:
            self.setFixedWidth(self.contents.minimumSizeHint().width())     
         
        return False
        
class DockWidget(QDockWidget):
    def __init__(self,widget):
        super(self.__class__, self).__init__()      
        self.setFloating(False)
        self.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.setWidget(widget) 
        
        self.default_title = None
        
    def setDefaultTitle(self,title):
        self.default_title = title
        self.setWindowTitle(title)
        
    def setTitle(self,title):
        self.setWindowTitle(self.default_title + title)
        

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

        