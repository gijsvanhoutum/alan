from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class ComboBox(QComboBox):
    clicked = pyqtSignal()

    def showPopup(self):
        self.clicked.emit()
        super(ComboBox, self).showPopup()
        
class CombiSetting(QGroupBox):

    command = pyqtSignal(str,list)
    
    def __init__(self,name,specs,parent=None):
        super(self.__class__, self).__init__(parent)

        self.name = name
        self.specs = specs   
        
        self.widgets = {}
        
        self._initialize()     
    
    def getName(self):
        return self.name

    def connectWidget(self,name,sort,sets):
        l1 = lambda data,n=name : self._sendCommand(data,n)
        l2 = lambda n=name : self._sendCommand("NONE",n)
            
        if sort == "C":
            widget = ComboBox()
            if len(sets) == 0:
                widget.clicked.connect(l2)
            else:
                widget.addItems(sets[0])
                widget.currentIndexChanged.connect(l1)
            
        elif sort == "S":
            widget = QSpinBox()  

            widget.setMinimum(sets[1])
            widget.setMaximum(sets[2])
            widget.setValue(sets[0]) 
            widget.valueChanged.connect(l1)
            
        elif sort == "L":
            widget = QLabel(name)
            
        elif sort == "T":
            widget = QCheckBox()
            widget.stateChanged.connect(l1)
            
        elif sort == "B":
            widget = QPushButton()
            
            if sets[0][-4:] == ".png":
                ic = QIcon("./icons/%s" % sets[0])
            else:
                ic = QIcon.fromTheme(sets[0])
            
            widget.setIcon(ic)
            widget.clicked.connect(l1)

        return widget
     
        
        
    def _initialize(self):     
        self.setTitle(self.name)
        policy = QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)                           
        self.setSizePolicy(policy)

        self.vbox = QVBoxLayout()
      
        for box_spec in self.specs:
            box = QGroupBox()
            box.setTitle(box_spec[0])

            grid = QGridLayout()

            for w in box_spec[1]:
                name = w[0]
                sort = w[1]
                pos = w[2:6]
                sets = w[6:]
                widget = self.connectWidget(name,sort,sets)
                grid.addWidget(widget,pos[0],pos[1],pos[2],pos[3]) 
                self.widgets[name] = widget   
                
            box.setLayout(grid)
            self.vbox.addWidget(box)
                
        self.setLayout(self.vbox)

    def _sendCommand(self,data,name):
        d = [name,data]      
        self.command.emit(self.name,d)