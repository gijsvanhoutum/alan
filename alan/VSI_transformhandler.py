from PyQt5.QtCore import QObject,pyqtSignal,pyqtSlot


class TransformHandler(QObject):

    ViewSignal = pyqtSignal(object) 

    def __init__(self,transform_classes, parent=None):
        super(self.__class__, self).__init__(parent)
 
        self.transform_classes = transform_classes        
        self.transforms = []
    
    def getInstNames(self):
        return [t.getName() for t in self.transforms]
        
    def getClassList(self):
        return self.transform_classes

    def getClassNames(self):
        return [t.getClassName() for t in self.transform_classes]
               
    @pyqtSlot(object)
    def processData(self,frame):
        [t.transform(frame) for t in self.transforms]
        """
        for t in self.transforms:
            name = t.getName()
            last_data = frame.getLastPassData()
            pass_data, view_data = t.transform(frame)
            frame.addData(name, pass_data, view_data)
        """
        # SEND THE UPDATED FRAME
        self.ViewSignal.emit(frame)
       
    def addTransform(self,cls_name):
        transform = self.createTransform(cls_name)
        self.transforms.append(transform)  
        return transform.getIVS()
        
    def createTransform(self,cls_name):
        for trf_cls in self.transform_classes:
            if trf_cls.getClassName() == cls_name:
                return trf_cls()        
    
    def removeTransform(self,index):
        self.transforms.pop(index-1)
        
    def getTransforms(self):
        return [t.getClassName() for t in self.transform_classes]
    
    def quitTransforms(self):
        self.transforms = []
        
    def doCommand(self,inst_name,par_dict):
        for t in self.transforms:
            if t.getName() == inst_name:
                t.setParams(par_dict)

             
        
        