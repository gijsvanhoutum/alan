import sys

from PyQt5.QtCore import QObject,pyqtSlot
from PyQt5.QtWidgets import QApplication

from gui import Gui
from viewhandler import ViewHandler
from view import View
from settinghandler import SettingHandler
from setting import CombiSetting
#from recordhandler import RecordHandler
#from record import Record
from devicehandler import DeviceHandler

from transformhandler import TransformHandler
from data import Frame

from device import *
from transform import *  


class ALAN(QObject):    
    """
    ALAN: Video Algorithm Analyzer.
    """
    def __init__(self):        
        super(self.__class__,self).__init__()      
        
        # Handles all what is displayed on the gui in terms of DataViews and
        # SettingViews.
        self.view_h = ViewHandler(View)

        # Handles all input widgets shown on the gui paired with the view
        self.setting_h = SettingHandler(CombiSetting)

        self.devices = [
            XirisAVI,
            InternshipAVI,
            # NEW_AVI,
            # KeyenceCSV,
            # MultiCSV,
            # XirisDAT
        ]
        
        self.device_h = DeviceHandler(self.devices,Frame)
        
        self.transforms = [
            SortBlue,
            Error,
            Normal,
            Triangle,
            Threshold,
            SegmentDeposition,
            SegmentLineEdge,
            ExtractWidth,
            ExtractHeight,
            WidthHeight,
            Complete,
            HW,
            W_Keyence,
            CompleteXiris,
            CompleteXiris2,
            CompleteXiris3,
            TriangleDirect, 
            SortThreshold, 
            TiltCorrection,
            MeltpoolThreshold,
            Calibration_Keyence,
            Intersection_Keyence,
            HoughLines
        ]
        # Handles transformation algorithms creation
        self.transform_h = TransformHandler(self.transforms)
        
        # The gui/controller sends signals when a action is activated. 
        # It needs the handlers for initialization.
        self.gui = Gui(
            self.view_h,
            self.setting_h,
            self.device_h,
            self.transform_h
        )

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
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ALAN()
    sys.exit(app.exec_())