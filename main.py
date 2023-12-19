# To access shell arguments given at application startup
import sys

# Gui library components
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

# Home made modules
from VSI_analyzer.VSI_gui import Gui
from VSI_analyzer.VSI_viewhandler import ViewHandler
from VSI_analyzer.VSI_view import View
from VSI_analyzer.VSI_settinghandler import SettingHandler
from VSI_analyzer.VSI_setting import CombiSetting
#from VSI_recordhandler import RecordHandler
#from VSI_record import Record
from VSI_analyzer.VSI_devicehandler import DeviceHandler

from VSI_analyzer.VSI_transformhandler import TransformHandler
from VSI_analyzer.VSI_data import Frame

from VSI_analyzer.VSI_device import *
from VSI_analyzer.VSI_transform import * 
from VSI_analyzer.VSI_model import Model

def main():

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
    
if __name__ == '__main__':
    main()
    

    
