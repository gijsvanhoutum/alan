import os 
import abc
# Second party libraries
import cv2               # OpenCV library
from PyQt5.QtCore import *  # PyQt core objects 

# Users defined classes / modules
from .VSI_metaclass import MetaClass # Metaclass for interfacing

import time
import csv
import pickle
import struct
import numpy as np
import pandas as pd

""" 
This class is a Abstract Base Class (Interface) for a device which shall
run in a seperate thread. The devicehandler will create and destroy 
devices. 

It should be reimplemented with the methods below 
without the classmethods.
""" 
    
class BaseDevice(QObject):
    __metaclass__  = MetaClass
                       
    __INST__ = 0             
    __NAME__ = None
    __ICON__ = None
    __SRC__  = None
    __TYPE__ = None
    __VIEW__ = None
    __SETS__ = None
    __HINT__ = None
    
    DeviceDataSignal = pyqtSignal(object)

    def __init__(self,data_class,parent=None):
        super(BaseDevice,self).__init__(parent)

        self.__INST__  +=1       
        self.name = self.__NAME__+"-"+str(self.__INST__)
        self.paths = self.getPaths()
        self.data_class = data_class
        self.initialize()      

    def getName(self):
        return self.name
        
    @abc.abstractmethod
    def initialize(self):
        pass
        
    @classmethod
    def getClassName(cls):
        return cls.__NAME__

    @classmethod        
    def getClassIcon(cls):
        return cls.__ICON__
    
    @classmethod        
    def getClassSource(cls):
        return cls.__SRC__
    
    @classmethod        
    def getClassType(cls):
        return cls.__TYPE__

    @classmethod         
    def getClassView(cls):
        return cls.__VIEW__

    @classmethod     
    def getClassSets(cls):
        return cls.__SETS__
    
    def getIVS(self):
        return self.name,self.__VIEW__,self.__SETS__
    
    def getPaths(self):
        paths = []

        for dev_name in os.listdir(self.__SRC__):
            if dev_name.startswith(self.__HINT__) or dev_name.endswith(self.__HINT__): 
                path = os.path.join(self.__SRC__,dev_name)

                if os.path.isfile(path):
                    paths.append(path)

        return paths
    

class NEW_AVI(BaseDevice):

    __NAME__ = "AVI - NEW"
    __ICON__ = "avi"
    __SRC__  = "./source/AVI_NEW/"
    __HINT__ = "avi"

    __VIEW__ = [["Video","c","r",0,0,6,8,{"I":"image"}]]               
    __SETS__ = [["Control",[["Start","B",0,0,1,1,"media-playback-start"],
                            ["Pause","B",1,0,1,1,"media-playback-pause"],
                            ["Stop","B",2,0,1,1,"media-playback-stop"],
                            ["Backward","B",0,1,1,1,"media-skip-backward"],
                            ["Forward","B",0,2,1,1,"media-skip-forward"],
                            ["FPS","L",1,1,1,1],
                            ["S2","S",1,2,1,1,10,1,99],
                            ["Loop","L",2,1,1,1],
                            ["C1","C",2,2,1,1,["On","Off"]]
                            ]
                            ]]

    def initialize(self):        
        
        self.path = 0
        self.loop = 0
        self.fps = 10

        self.timer = self.setupTimer()
        self.reader = self.setupReader()

    def setupTimer(self):
        timer = QTimer()
        timer.setInterval(int(1000/self.fps))
        timer.timeout.connect(self.sendFrame)     
        return timer

    def setupReader(self):
        return cv2.VideoCapture(self.paths[self.path])

    def sendFrame(self):
        retval,image = self.reader.read()
        
        if retval:
            self.pos = self.reader.get(1)
            frame = self.data_class() 
            frame.setPosition(self.pos)   
            epoch = struct.unpack("d",image[:8,0,0].tobytes())[0]

            s = image.shape
            cols = s[1]/3
            rgb = np.zeros((s[0],int(cols)-1,3),dtype=np.uint8)
            rgb[:,:,:] = np.moveaxis(np.array(np.hsplit(image[:,:,2],3)),0,-1)[:,1:,:]
            RGB = cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
         
            pass_data = [RGB]
            view_data = {"I":RGB}
                
            frame.addData(self.name,pass_data, view_data)
            self.DeviceDataSignal.emit(frame)            
        else:
            try:
                self.path+=1
                self.reader = self.setupReader()
            except:
                self.path = 0
                if self.loop == 1:
                    self.timer.stop()
                    
                self.reader = self.setupReader()

    def setPosition(self,diff):
        self.timer.stop()          
        self.reader.set(1,self.pos-1+diff)    
        self.sendFrame()
        
    def setParams(self,par_list):
        name = par_list[0]
        value = par_list[1] 
        
        if name == 'Start':
            self.timer.start()
            self.updateTime = time.time()
        elif name == 'Pause':
            self.timer.stop()
        elif name == 'Stop':
            self.timer.stop()
            self.setupReader()
        elif name == 'Backward':
            self.setPosition(-1)           
        elif name == 'Forward':
            self.setPosition(1)           
        elif name == "S2":
            self.timer.setInterval(int(1000/value))
        elif name == 'C1':
            self.loop = value
            
class InternshipAVI(BaseDevice):

    __NAME__ = "AVI - Internship"
    __ICON__ = "avi"
    __SRC__  = "./source/AVI_internship/"
    __HINT__ = "avi"

    __VIEW__ = [["Video","c","r",0,0,6,8,{"I":"image"}]]               
    __SETS__ = [["Control",[["Start","B",0,0,1,1,"media-playback-start"],
                            ["Pause","B",1,0,1,1,"media-playback-pause"],
                            ["Stop","B",2,0,1,1,"media-playback-stop"],
                            ["Backward","B",0,1,1,1,"media-skip-backward"],
                            ["Forward","B",0,2,1,1,"media-skip-forward"],
                            ["FPS","L",1,1,1,1],
                            ["S2","S",1,2,1,1,10,1,99],
                            ["Loop","L",2,1,1,1],
                            ["C1","C",2,2,1,1,["On","Off"]]
                            ]
                            ]]

    def initialize(self):        
        
        self.path = 0
        self.loop = 0
        self.fps = 10

        self.timer = self.setupTimer()
        self.reader = self.setupReader()
        #self.record = self.getRecord()
        #self.times = self.record.getTimes()

    def setupTimer(self):
        timer = QTimer()
        timer.setInterval(int(1000/self.fps))
        timer.timeout.connect(self.sendFrame)     
        return timer

    def setupReader(self):
        return cv2.VideoCapture(self.paths[self.path])

    def getRecord(self):
        return pickle.load(open(self.paths[self.path]+self.device_file_ext,"rb"))  
            
    def sendFrame(self):
        retval,image = self.reader.read()
        
        if retval:
            pass_data = [image]
            view_data = {"I":image}

            self.pos = self.reader.get(1)
            frame = self.data_class() 
            frame.setPosition(self.pos)   
            #time = self.times[self.pos]  
            frame.addData(self.name,pass_data, view_data)
            self.DeviceDataSignal.emit(frame)            
        else:
            try:
                self.path+=1
                self.reader = self.setupReader()
                #self.record = self.getRecord()
                #self.times = self.record.getTimes()
            except:
                self.path = 0
                if self.loop == 1:
                    self.timer.stop()
                    
                self.reader = self.setupReader()
                #self.record = self.getRecord()
                #self.times = self.record.getTimes()

    def setPosition(self,diff):
        self.timer.stop()          
        self.reader.set(1,self.pos-1+diff)    
        self.sendFrame()
        
    def setParams(self,par_list):
        name = par_list[0]
        value = par_list[1] 
        
        if name == 'Start':
            self.timer.start()
            self.updateTime = time.time()
        elif name == 'Pause':
            self.timer.stop()
        elif name == 'Stop':
            self.timer.stop()
            self.setupReader()
        elif name == 'Backward':
            self.setPosition(-1)           
        elif name == 'Forward':
            self.setPosition(1)           
        elif name == "S2":
            self.timer.setInterval(int(1000/value))
        elif name == 'C1':
            self.loop = value
            

class XirisAVI(BaseDevice):

    __INST__ = 0
    __NAME__ = "AVI - Xiris"
    __ICON__ = "avi"
    __TYPE__ = "file"
    __SRC__  = "./source/AVI_xiris/"
    __HINT__ = "avi"
    
    __VIEW__ = [["Data","c","r",0,0,3,4,{"I":"image"}]]
                 
    __SETS__ = [["Control",[["Start","B",0,0,1,1,"media-playback-start"],
                            ["Pause","B",1,0,1,1,"media-playback-pause"],
                            ["Stop","B",2,0,1,1,"media-playback-stop"],
                            ["Backward","B",0,1,1,1,"media-skip-backward"],
                            ["Forward","B",0,2,1,1,"media-skip-forward"],
                            ["FPS","L",1,1,1,1],
                            ["FPS_S","S",1,2,1,1,10,1,1000]
                            ]
                            ]
                ]

    def initialize(self):        
        self.path = 0
        self.fps = 10.0        

        self.setupReader(self.path)
        self.setupTimer(self.fps)

    def sendFrame(self):
        retval,image = self.reader.read()
        
        if retval:
            pass_data = [image]
            view_data = {"I":image}    
            frame = self.data_class()                     
            frame.addData(self.name,pass_data, view_data)
            self.DeviceDataSignal.emit(frame)  
        else:
            if self.paths[self.path] == self.paths[-1]:
                self.path = 0
            else:
                self.path+=1
                
            self.setupReader(self.path)
        
    def setupReader(self,index):
        self.reader = cv2.VideoCapture(self.paths[index])
        
    def setupTimer(self,fps):
        self.timer = QTimer()
        self.timer.setInterval(int(1000.0/fps))
        self.timer.timeout.connect(self.sendFrame)            

    def setParams(self,par_list):
        name = par_list[0]
        value = par_list[1] 
        
        if name == 'Start':
            self.timer.start()
        elif name == 'Pause':
            self.timer.stop()
        elif name == 'Stop':
            self.timer.stop()
            self.path = 0
            self.setupReader(self.path)    
        elif name == "Backward":
            self.pos = self.reader.get(1)   
            print(self.pos)
            self.reader.set(1,int(self.pos-2))    
            self.sendFrame()
            print(self.reader.get(1))
        elif name == "Forward":
            self.pos = self.reader.get(1)        
            self.reader.set(1,self.pos+1)    
            self.sendFrame()        
        elif name == "FPS_S":
            self.timer.setInterval(int(1000/value)) 

            
class XirisDAT(BaseDevice):

    __INST__ = 0
    __NAME__ = "DAT - Xiris"
    __ICON__ = "dat"
    __TYPE__ = "file"
    __SRC__  = "./source/DAT_xiris/dat_4"
    __HINT__ = "dat"

    __VIEW__ = [["Xiris DAT Original","c","r",0,0,3,4,
                 {"I":"image",
                  "top":"hline",
                  "bottom":"hline",
                  "left":"vline",
                  "right":"vline"}]]#,
                 
    __SETS__ = [["Control",[["Start","B",0,0,1,1,"media-playback-start"],
                            ["Pause","B",1,0,1,1,"media-playback-pause"],
                            ["Stop","B",2,0,1,1,"media-playback-stop"],
                            ["Backward","B",0,1,1,1,"media-skip-backward"],
                            ["Forward","B",0,2,1,1,"media-skip-forward"],
                            ["FPS","L",1,1,1,1],
                            ["S2","S",1,2,1,1,10,1,1000],
                            ["Loop","L",2,1,1,1],
                            ["C1","C",2,2,1,1,["On","Off"]]
                            ]
                            ],
                ["ROI",[["Top","L",0,0,1,1],
                        ["Down","L",1,0,1,1],
                        ["top","S",0,1,1,1,0,0,5000],
                        ["down","S",1,1,1,1,0,0,5000],
                        ["Left","L",2,0,1,1],
                        ["Right","L",3,0,1,1],
                        ["left","S",2,1,1,1,0,0,5000],
                        ["right","S",3,1,1,1,0,0,5000],
                        ]
                        ],
                ]

    def initialize(self):        
        self.device_dir   = os.path.join(self.__SRC__)
        self.device_hint  = ".dat"
        
        self.path = 0
        self.loop = 0

        self.setupTimer()
        
        self.fps = 0
        
        self.top = 0
        self.down = 0
        self.left = 0
        self.right = 0
        
        self.pause = False

    def get_image(self,file_name):
        
        s = np.fromfile(file_name, dtype="uint8",count=2)
        
        img = {}    
        if max(s) == 0:  # shcmea 1
            h = np.fromfile(file_name, dtype="uint32",count=9)    
            img["SCHEMA"] = 1 if h[0] else 0
            img["HEADER_LEN"] = h[1]
            img["AOI_LEFT"] = h[2]
            img["AOI_TOP"] = h[3]
            img["AOI_RIGHT"] = h[4]
            img["AOI_BOTTOM"] = h[5]
            img["IMG_WIDTH"] = h[6]
            img["BITS_PP"] = h[7]
            img["PIXEL_FMT"] = h[8]
            img["IMG_HEIGHT"] = img["AOI_BOTTOM"] - img["AOI_TOP"]
            img["DATA_START"] = 36
        else: # schema 0
            h = np.fromfile(file_name, dtype="uint16",count=4) 
            img["SCHEMA"] = 0         
            img["IMG_HEIGHT"] = h[0]
            img["IMG_WIDTH"] = h[1]
            img["BITS_PP"] = h[3]
            img["PIXEL_FMT"] = h[4] 
            img["DATA_START"] = 9   
        
        if img["PIXEL_FMT"] == 0:
            d = np.fromfile(file_name, dtype="uint8")[img["DATA_START"]:]
            img["DATA"] = d.reshape((img["IMG_HEIGHT"],img["IMG_WIDTH"]))
        elif img["PIXEL_FMT"] == 1:
            data = np.zeros((img["IMG_HEIGHT"]*img["IMG_WIDTH"]),dtype=np.uint16)
            d = np.fromfile(file_name, dtype="uint8")[img["DATA_START"]:]
            d = np.unpackbits(d)
            d = d.reshape(int(d.shape[0]/12),12)
            rows = np.arange(1,data.shape[0],2)
            d[rows,:] = np.roll(d[rows,:],-4,axis=1)
            for i in np.arange(0,12):
                data+=2**i*d[:,11-i]
                
            img["DATA"] = data.reshape((img["IMG_HEIGHT"],img["IMG_WIDTH"]))  
    
        return img            
    
    def sendFrame(self):
        image_dict = self.get_image(self.paths[self.path])
        I = image_dict["DATA"]

        s = I.shape
        G = I[self.top:s[0]-self.down,self.left:s[1]-self.right]

        if not self.pause:
            
            self.path+=1
            if self.path == len(self.paths):
                self.path = 0
            
        frame = self.data_class() 
        frame.setPosition(self.path)
        pass_data = [G]
        view_data = {"I":I,
                     "top":self.top,
                     "bottom":s[0]-self.down,
                     "left":self.left,
                     "right":s[1]-self.right}
                     
        frame.addData(self.name,pass_data, view_data)
        self.DeviceDataSignal.emit(frame)          
        
    def setupTimer(self):
        self.timer = QTimer()
        self.timer.setInterval(int(1000/10))
        self.timer.timeout.connect(self.sendFrame)            
    
    def setPosition(self,diff): 
        self.pos = self.pos-1+diff
        self.sendFrame()
        
    def setParams(self,par_list):
        name = par_list[0]
        value = par_list[1] 
        
        if name == 'Start':
            self.pause = False
            self.timer.start()
        elif name == 'Pause':
            self.pause = True
        elif name == 'Stop':
            self.timer.stop()
            self.setupReader()
        elif name == 'Backward':
            self.setPosition(-1)           
        elif name == 'Forward':
            self.setPosition(1)           
        elif name == "S2":
            self.timer.setInterval(int(1000/value))
        elif name == 'C1':
            self.loop = value    
        elif name == "top":
            self.top = value
        elif name == "down":
            self.down = value
        elif name == "left":
            self.left = value
        elif name == "right":
            self.right = value
            
class KeyenceCSV(BaseDevice):

    __INST__ = 0
    __NAME__ = "CSV - Keyence"
    __ICON__ = "csv"
    __TYPE__ = "file"
    __HINT__ = "csv"
    
    __SRC__ = "./source/CSV_keyence/"

    __VIEW__ = [["Row","height","r",0,0,3,4,{"I":"image"}]]
    __SETS__ = [["Control",[["Start","B",0,0,1,1,"media-playback-start"],
                            ["Pause","B",1,0,1,1,"media-playback-pause"],
                            ["Stop","B",2,0,1,1,"media-playback-stop"],
                            ["Backward","B",0,1,1,1,"media-skip-backward"],
                            ["Forward","B",0,2,1,1,"media-skip-forward"],
                            ["FPS","L",1,1,1,1],
                            ["S2","S",1,2,1,1,10,1,99],
                            ["Loop","L",2,1,1,1],
                            ["C1","C",2,2,1,1,["On","Off"]]
                            ]
                            ]]

    def initialize(self):        
        self.device_dir   = os.path.join(self.__SRC__)
        self.device_hint  = ".csv"
        
        self.path = 0
        self.loop = 0

        self.setupTimer()
        self.csv_dict = self.csv_to_dict(self.paths[self.path])
        
        self.fps = 0
        self.pos = 0

    def csv_to_dict(self,filepath):
        d = {}  
        with open(filepath, 'r') as f:
            reader = csv.reader(f, delimiter=',')
    
            go = True
            for i, row in enumerate(reader):
                if go:
                    if i<46:
                        if len(row) > 1:
                            key = row[0]
                            value = row[1]
                            if len(row) > 2:       
                                unit = row[2]
                            else:
                                unit = None
                                
                            d[key] = [value,unit]
                    elif i==49:
                        d["Width"] = int(row[0])
                        d["Height"] = int(row[1])
                        d["Skip Amount"] = int(row[2])
                    elif i>51:
                        arr = np.zeros((d["Height"],d["Width"]))
                        
                        for i,row in enumerate(reader):
                            if len(row) == d["Width"]:
                                arr[i,:] = list(map(int, row))
                            else:
                                go = False
                        d["Data"] = arr
                        
        return d
                    
    def sendFrame(self):
        frame = self.data_class() 
        pass_data = [self.csv_dict]
        view_data = {"I":self.csv_dict["Data"]}
                     
        frame.addData(self.name,pass_data, view_data)
        self.DeviceDataSignal.emit(frame) 
                  
    def setupTimer(self):
        self.timer = QTimer()
        self.timer.setInterval(1000/10)
        self.timer.timeout.connect(self.sendFrame)            

    def setParams(self,par_list):
        name = par_list[0]
        value = par_list[1] 
        print(name,value)
        if name == 'Start':
            self.timer.start()
        elif name == 'Pause':
            self.timer.stop()
        elif name == 'Stop':
            self.timer.stop()
            self.setupReader()
        elif name == 'Backward':
            self.setPosition(-1)           
        elif name == 'Forward':
            self.setPosition(1)           
        elif name == "S2":
            self.timer.setInterval(1000/value)
        elif name == 'C1':
            self.loop = value  

class MultiCSV(BaseDevice):

    __INST__ = 0
    __NAME__ = "CSV - Multi"
    __ICON__ = "csv"
    __TYPE__ = "file"
    __HINT__ = "csv"
    
    __SRC__ = "./source/CSV_multi/"

    __VIEW__ = [["Algorithm","c","r",0,0,6,8,
                            {"algo_height":"curve",
                             "algo_width":"curve"}],
                ["Keyence","c","r",6,0,6,8,
                            {"key_height":"curve",
                             "key_width":"curve"}],
                            ]
    
    __SETS__ = [["Control",[["Start","B",0,0,1,1,"media-playback-start"],
                            ["Pause","B",1,0,1,1,"media-playback-pause"],
                            ["Stop","B",2,0,1,1,"media-playback-stop"],
                            ["Backward","B",0,1,1,1,"media-skip-backward"],
                            ["Forward","B",0,2,1,1,"media-skip-forward"],
                            ["FPS","L",1,1,1,1],
                            ["S2","S",1,2,1,1,10,1,99],
                            ["Loop","L",2,1,1,1],
                            ["C1","C",2,2,1,1,["On","Off"]]
                            ]
                            ]]

    def initialize(self):        
        self.device_dir   = os.path.join(self.__SRC__)
        self.device_hint  = ".csv"
        
        self.path = 0
        self.loop = 0       
        self.fps = 10
        self.pos = 0
        
        self.setupTimer(self.fps)
            
    def sendFrame(self):
        
        algo = self.dataframes[0].values
        key = self.dataframes[1].values

        pass_data = [self.dataframes]
        
        view_data = {"algo_height":[np.arange(algo.shape[0]),algo[:,0]],
                     "algo_width":[np.arange(algo.shape[0]),algo[:,1]],
                     "key_height":[np.arange(key.shape[0]),key[:,0]],
                     "key_width":[np.arange(key.shape[0]),key[:,1]]}

        frame = self.data_class()                      
        frame.addData(self.name,pass_data, view_data)
        self.DeviceDataSignal.emit(frame) 
        
        self.pos +=1
                  
    def setupTimer(self,fps):
        self.timer = QTimer()
        self.timer.setInterval(int(1000/fps))
        self.timer.timeout.connect(self.sendFrame)            

    def setParams(self,par_list):
        name = par_list[0]
        value = par_list[1] 

        if name == 'Start':
            self.timer.start()
        elif name == 'Pause':
            self.timer.stop()
        elif name == 'Stop':
            self.timer.stop()
            self.setupReader()
        elif name == 'Backward':
            self.setPosition(-1)           
        elif name == 'Forward':
            self.setPosition(1)           
        elif name == "S2":
            self.timer.setInterval(int(1000/value))
        elif name == 'C1':
            self.loop = value  

        
