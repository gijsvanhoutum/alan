# To access shell arguments given at application startup
import sys

# Gui library components
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

# Home made modules
from VSI_gui import Gui
from VSI_viewhandler import ViewHandler
from VSI_view import View
from VSI_settinghandler import SettingHandler
from VSI_setting import CombiSetting
#from VSI_recordhandler import RecordHandler
#from VSI_record import Record
from VSI_devicehandler import DeviceHandler

from VSI_transformhandler import TransformHandler
from VSI_data import Frame

from VSI_device import *
from VSI_transform import *  



class Model(QObject):    
    # Initialize program, add all available classes to the handlers and 
    # connect all signals.
    def __init__(self,gui,
                 viewhandler,view,
                 settinghandler,setting,
                 devicehandler,devices,
                 transformhandler,transforms,
                 data
                 ):
                     
        # Initialize Super Class to be able to connect Signals            
        super(self.__class__,self).__init__()        
        # Handles all what is displayed on the gui in terms of DataViews and
        # SettingViews.
        self.view_h = viewhandler(view)

        # Handles all input widgets shown on the gui paired with the view
        self.setting_h = settinghandler(setting)

        self.device_h = devicehandler(devices,data)
        # Handles transformation algorithms creation
        self.transform_h = transformhandler(transforms)
        # The gui/controller sends signals when a action is activated. 
        # It needs the handlers for initialization.
        self.gui = gui(self.view_h,self.setting_h,self.device_h,
                       self.transform_h)

        # Connect all signals with slots
        self._connectSignals()

    # To connect the Gui and Handlers with eachother these Signal-Slot 
    # connections are made.
    def _connectSignals(self): 
        self.gui.generator.connect(self._openDevice)
        self.gui.transform.connect(self._addTransform)
        self.view_h.command.connect(self._doViewCommand)
        self.setting_h.command.connect(self._doSettingCommand)
        self.device_h.DataSignal.connect(self.transform_h.processData)
        self.transform_h.ViewSignal.connect(self.view_h.updateViews)
        #self.transform_h.DataSignal.connect(self.record_h.addFrame)
                   
    # If the available device in the Device-Menu is clicked this function
    # is called. It opens up the device and initializes the Gui with the 
    # right Setting and View widgets.
    @pyqtSlot(str)
    def _openDevice(self,cls_name):
        # i,v,s returns the Instance name, View specs & Setting specs
        i,v,s = self.device_h.setDevice(cls_name) 
        self.view_h.addView(i,v)  
        self.setting_h.addSetting(i,s) 

    # If the available Transformation in the Gui-ComboBox is clicked this 
    # function is called. It adds the Transformation to its Handler and 
    # sets up the Gui with its5 paired View and Setting widgets.
    @pyqtSlot(str)
    def _addTransform(self,cls_name):
        # i,v,s returns the Instance name, View specs & Setting specs
        i,v,s = self.transform_h.addTransform(cls_name)         
        self.view_h.addView(i,v)  
        self.setting_h.addSetting(i,s)             
    
    # The View Handler has close buttons on each Tab. When pressed the index
    # is passed or when it is the last Tab -1 is passed so that the device
    # can be Quit.
    @pyqtSlot(int)
    def _doViewCommand(self,command):
        self.setting_h.removeSetting(command)
        if command == 0:
            self.device_h.quitDevice()
            self.gui.initial()
        else:
            self.transform_h.removeTransform(command)
            
    # When a value in a setting widget is changed it needs to be send to the 
    # right transform or device. The instance name and parameters are send.
    @pyqtSlot(str,list)
    def _doSettingCommand(self,inst_name,par_list):
        if inst_name == self.device_h.getInstName():
            self.device_h.doCommand(inst_name,par_list)
        elif inst_name in self.transform_h.getInstNames():
            self.transform_h.doCommand(inst_name,par_list)
        
    # Start the Gui
    def start(self):
        self.gui.show()
        
if __name__ == '__main__':
    a = QApplication(sys.argv)


    Sources = [NEW_AVI,InternshipAVI,XirisAVI,KeyenceCSV,MultiCSV,XirisDAT]
    Algos = [SortBlue,Error,Normal,Triangle,Threshold,SegmentDeposition, \
                  SegmentLineEdge,ExtractWidth,ExtractHeight,WidthHeight, \
                  Complete,HW,W_Keyence,CompleteXiris,CompleteXiris2, \
                  CompleteXiris3,TriangleDirect, SortThreshold, TiltCorrection, \
                  MeltpoolThreshold,Calibration_Keyence,Intersection_Keyence,HoughLines]
                                    
    p = Model(Gui,
              ViewHandler,View,
              SettingHandler,CombiSetting,
              DeviceHandler,Sources,
              TransformHandler,Algos,
              Frame
              )
                          
    p.start()
    a.exec_()
    