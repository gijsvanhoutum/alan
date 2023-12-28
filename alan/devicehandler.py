from PyQt5.QtCore import QObject,QThread,pyqtSignal


class DeviceHandler(QObject):

    DataSignal = pyqtSignal(object)

    def __init__(self,device_classes,data_class,parent=None):
        super(self.__class__, self).__init__(parent)
    
        self.device_classes = device_classes
        self.data_class = data_class   
        
        self.device_used = None 
        self.status = "unset"

    def setDevice(self,name):
        if self.status != "unset":
            self.quitDevice()
        
        self.device_thread = QThread()
        self.device_used = self.createDevice(name)
        self.device_used.moveToThread(self.device_thread)
        self.device_used.DeviceDataSignal.connect(self.DataSignal)
        self.device_thread.start()
        self.status = "set"

        return self.device_used.getIVS()

    def createDevice(self,cls_name):
        for cls in self.device_classes:
            if cls.getClassName() == cls_name:
                return cls(self.data_class)  
                    
    def getClassList(self):
        return self.device_classes
 
    def getInstName(self):
        return self.device_used.getName()
        
    def getClassNames(self):
        return [d.getClassName() for d in self.device_classes]
    
    def getUsedDeviceInfo(self):
        return self.device_used.getInfo()
        
    def getUsedNamePath(self):
        if self.status == "unset":
            return None
        else:
            return self.device_used.getNamePath()
            
    def startDevice(self):
        if self.status != "run":
            self.device_used.start()
            self.status = "run"
                
    def stopDevice(self):
        self.device_used.setParams(["Stop",0])
        self.status = "stop"
            
    def quitDevice(self):
        self.device_used.setParams(["Pause",0])
        self.quitThread()
        self.status = "unset"
    
    def quitThread(self):
        self.device_thread.quit()
        self.device_thread.wait() 
                         
    def getUsedDeviceName(self):
        if self.status == "unset":
            return None
        else:
            return self.device_used.getDeviceName()
       
    def hasDevice(self):
        if self.status == "unset":
            return False
        else:
            return True
            
    def doCommand(self,inst_name,par_list):
        self.device_used.setParams(par_list)
            