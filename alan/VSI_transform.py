from abc import ABCMeta,abstractmethod
import cv2
import numpy as np
import csv
# from scipy import ndimage
import math

from VSI_functions import *

# import scipy.ndimage.measurements as sp

# from skimage.segmentation import watershed
# from skimage.feature import peak_local_max
# from scipy import ndimage
# import scipy

# from sklearn.cluster import DBSCAN
# from sklearn.datasets import make_blobs

# from scipy.optimize import minimize_scalar,curve_fit

# from scipy.stats import norm


class Transformation:
    __metaclass__ = ABCMeta
    
    """ 
    This class is a Abstract Base Class (Interface) for a transformation.
    
    It should be reimplemented with the methods below.
    """ 
    __VIEW__ = None
    __SETS__ = None
    
    def __init__(self):
        self.__class__.__INST__  +=1       
        self.name = self.__NAME__+"-"+str(self.__INST__)

        self.initialize()        
        #self.initSETS()

    def getIVS(self):
        return self.name,self.__VIEW__,self.__SETS__
        
    def getName(self):
        return self.name
        
    def setParams(self,l_params):
        setattr(self,l_params[0],l_params[1])
            
    @classmethod        
    def getClassName(self):
        return self.__NAME__
        
    @classmethod
    def getViewSpecs(self):
        return self.__VIEW__
    
    def initSETS(self):
        if self.__SETS__ is not None:
            if type(self.__SETS__) is list:
                for l_box in self.__SETS__:
                    if type(l_box) is list:
                        for l_entry in l_box[1]:
                            if type(l_entry) is list:
                                if l_entry[1] != "L":
                                    s_name = l_entry[0]
                                    value = l_entry[6]
            
                                    if type(value) is list:
                                        val = 0
                                    else:
                                        val = value
                                                                  
                                    setattr(self,s_name,val)
                            else:
                                raise TypeError("ENTRY in BOX in __SETS__ fault. " \
                                      "Should be a LIST object")
                    else:
                        raise TypeError("BOX in __SETS__ fault. Should be a LIST object")
            else:
                raise TypeError("__SETS__ fault. Should be a LIST object")
        
    @abstractmethod    
    def transform(self,frame):
        """
        This method takes a frame and needs to be reimplemented.
        
        The frame consists out of the original 
        generated data and out of data produced as 
        outputs from previous transformations.
        """

class SortBlue(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Sort Blue"                                  
    __VIEW__ = [["Thresholded","c","r",0,0,6,8,
                            {"T":"image"}],
                ["Histogram","c","r",6,0,3,8,
                            {"H":"curve",
                             "S":"curve",
                             "V":"curve"}]
                            ]
                
    __SETS__ = [["Threshold",[["H","L",0,0,1,1],
                              ["hl","S",0,1,1,1,0,0,5000],
                              ["hh","S",0,2,1,1,0,0,5000],
                              ["S","L",1,0,1,1],
                              ["sl","S",1,1,1,1,0,0,5000],
                              ["sh","S",1,2,1,1,0,0,5000],
                              ["V","L",2,0,1,1],
                              ["vl","S",2,1,1,1,0,0,5000],
                              ["vh","S",2,2,1,1,0,0,5000],
                             ]
                             ],
                ]
                    
    def initialize(self):    
        self.hl = 0
        self.hh = 255
        self.sl = 0
        self.sh = 255
        self.vl = 0
        self.vh = 255

    def transform(self,frame):
        device_data = frame.getLastPassData()
        
        I = device_data[0]
        HSV = I#cv2.cvtColor(I,cv2.COLOR_RGB2HSV)
        H = cv2.calcHist([HSV],[0],None,[256],[0,256]).flatten()
        S = cv2.calcHist([HSV],[1],None,[256],[0,256]).flatten()
        V = cv2.calcHist([HSV],[2],None,[256],[0,256]).flatten()

        MIN = np.array([self.hl,self.sl,self.vl],np.uint8)
        MAX = np.array([self.hh,self.sh,self.vh],np.uint8)
        mask = cv2.inRange(HSV,MIN,MAX)
        T = cv2.bitwise_and(I,I,mask=mask)
        rng = np.arange(len(H))

        view_data = {"T":T,
                     "H":[rng,H],
                     "S":[rng,S],
                     "V":[rng,V]
                     }
                     
        
        pass_data = [I]
        
        frame.addData(self.getName(),pass_data,view_data)
    
    def setParams(self,par_list):
        name = par_list[0]
        value = par_list[1] 
        
        if name == "hl":
            self.hl = value
        elif name == "hh":
            self.hh = value
        elif name == "sl":
            self.sl = value
        elif name == "sh":
            self.sh = value
        elif name == "vl":
            self.vl = value
        elif name == "vh":
            self.vh = value


class SortGray(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Sort Gray"                                  
    __VIEW__ = [["Thresholded","c","r",0,0,6,8,
                            {"T":"image"}],
                ["Histogram","c","r",6,0,3,8,
                            {"H":"curve",
                             "S":"curve",
                             "V":"curve"}]
                            ]
                
    __SETS__ = [["Threshold",[["H","L",0,0,1,1],
                              ["hl","S",0,1,1,1,0,0,5000],
                              ["hh","S",0,2,1,1,0,0,5000],
                              ["S","L",1,0,1,1],
                              ["sl","S",1,1,1,1,0,0,5000],
                              ["sh","S",1,2,1,1,0,0,5000],
                              ["V","L",2,0,1,1],
                              ["vl","S",2,1,1,1,0,0,5000],
                              ["vh","S",2,2,1,1,0,0,5000],
                             ]
                             ],
                ]
                    
    def initialize(self):    
        self.hl = 0
        self.hh = 255
        self.sl = 0
        self.sh = 255
        self.vl = 0
        self.vh = 255

    def transform(self,frame):
        device_data = frame.getLastPassData()
        
        I = device_data[0]
        HSV = I#cv2.cvtColor(I,cv2.COLOR_RGB2HSV)
        H = cv2.calcHist([HSV],[0],None,[256],[0,256]).flatten()
        S = cv2.calcHist([HSV],[1],None,[256],[0,256]).flatten()
        V = cv2.calcHist([HSV],[2],None,[256],[0,256]).flatten()

        MIN = np.array([self.hl,self.sl,self.vl],np.uint8)
        MAX = np.array([self.hh,self.sh,self.vh],np.uint8)
        mask = cv2.inRange(HSV,MIN,MAX)
        T = cv2.bitwise_and(I,I,mask=mask)
        rng = np.arange(len(H))

        view_data = {"T":T,
                     "H":[rng,H],
                     "S":[rng,S],
                     "V":[rng,V]
                     }
                     
        
        pass_data = [I]
        
        frame.addData(self.getName(),pass_data,view_data)
    
    def setParams(self,par_list):
        name = par_list[0]
        value = par_list[1] 
        
        if name == "hl":
            self.hl = value
        elif name == "hh":
            self.hh = value
        elif name == "sl":
            self.sl = value
        elif name == "sh":
            self.sh = value
        elif name == "vl":
            self.vl = value
        elif name == "vh":
            self.vh = value
            
        
class Error(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Error"                                  
    __VIEW__ = [["(a) Channel","c","r",0,0,6,8,
                            {"I":"image",
                             "$\delta^{left}_{ini}$":"vline",
                             "$\delta^{right}_{ini}$":"vline"}],
                ["(b) Squared Errors","c","r",0,8,6,8,
                            {"SE":"image",
                             "$\delta^{left}_{ini}$":"vline",
                             "$\delta^{right}_{ini}$":"vline"}],
                ["(c) Template","r","pixel value",6,0,3,8,
                            {"$T$":"curve"}],
                ["(d) Sum-of-Squared Errors","c","SSE",6,8,3,8,
                            {"$SSE$":"curve",
                             "$\delta^{left}_{ini}$":"vline",
                             "$\delta^{right}_{ini}$":"vline"}],
                             ]
                              
    def initialize(self):    
        pass

    def transform(self,frame):
        pass_data = frame.getLastPassData()
        img = pass_data[0]           
        l = 130
        r = img.shape[1]-130

          
        R,G,B = cv2.split(img)
        
        channels = [R,G,B]        
        squared = []
        templates = []
        ssqs = []
        
        for ch in channels:           
            left_cols = ch[ :, :l ]
            right_cols = ch[ :, r: ] 
            total_cols = np.hstack( ( left_cols, right_cols ) )    
            template = np.mean( total_cols, axis=1)

            error = ( ch.T - template).T  
            square = error**2
            ssq = np.sum(square, axis=0)
            
            templates.append(template)
            squared.append(square)
            ssqs.append(ssq)

        ch_nr = 1

        view_data = {"I":channels[ch_nr],
                     "$T$":[np.arange(len(templates[ch_nr])),templates[ch_nr]],
                     "SE":squared[ch_nr],
                     "$SSE$":[np.arange(len(ssqs[ch_nr])),ssqs[ch_nr]],
                     "$\delta^{left}_{ini}$":l,
                     "$\delta^{right}_{ini}$":r,
                              }
                     
        pass_data = [img,ssqs,l,r]
        
        frame.addData(self.getName(),pass_data,view_data)
        
class Normal(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Normal"                                  
    __VIEW__ = [["(a) RGB Image","c","r",0,0,12,16,
                            {"$I$":"image",
                             "$\delta^{left}_{ini}$":"vline",
                             "$\delta^{right}_{ini}$":"vline"}],
                ["(b) Sum-of-Squared Errors","c","SSE",0,16,9,16,
                            {"$SSE_{c1}$":"curve",
                             "$SSE_{c2}$":"curve",
                             "$SSE_{c3}$":"curve",
                             "$\delta^{left}_{ini}$":"vline",
                             "$\delta^{right}_{ini}$":"vline"}],
                ["(c) Normalized","c","NSSE",9,16,9,16,
                            {"$NSSE_{c1}$":"curve",
                             "$NSSE_{c2}$":"curve",
                             "$NSSE_{c3}$":"curve",
                             "$\delta^{left}_{ini}$":"vline",
                             "$\delta^{right}_{ini}$":"vline"}],
                ["(d) Total Error","c","TE",12,0,6,16,
                            {"$TE_{c}$":"curve",
                             "$\delta^{left}_{ini}$":"vline",
                             "$\delta^{right}_{ini}$":"vline"}],
                             ]
                                 
    def initialize(self):    
        pass

    def transform(self,frame):
        pass_data = frame.getLastPassData()

        img = pass_data[0]         
        sses = pass_data[1]
        l = pass_data[2]
        r = pass_data[3]
        
        nsses = []
        s = []
        s_n = []
        for sse in sses:  
            sides = np.hstack((sse[:l],sse[r:]))
            std = np.std( sides, ddof=1)  
            mn = np.mean( sides )             
            norm = (sse - mn) / std
            norm_side = np.hstack((norm[:l],norm[r:]))
            nsses.append(norm)
            
            s_n.append(norm_side)
            s.append(sides)
            
            
        rng_err, h_err = histograms(s)
        rng_nrm, h_nrm = histograms(s_n)
        
        te = euclidean(nsses)

     
        rng = np.arange(img.shape[1])
        

        view_data = {"$I$":img,
                     "$SSE_{c1}$":[rng,sses[0]],
                     "$SSE_{c2}$":[rng,sses[1]],
                     "$SSE_{c3}$":[rng,sses[2]],
                     "$NSSE_{c1}$":[rng,nsses[0]],
                     "$NSSE_{c2}$":[rng,nsses[1]],
                     "$NSSE_{c3}$":[rng,nsses[2]],
                     "$TE_{c}$":[rng,te],
                     "$\delta^{left}_{ini}$":l,
                     "$\delta^{right}_{ini}$":r,
                              }
                     
        pass_data = [img,te,l,r]
        
        frame.addData(self.getName(),pass_data,view_data)
        
class Triangle(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Triangle"                                  
    __VIEW__ = [["(a) RGB Image","c","r",0,0,6,8,
                            {"$I$":"image",
                             "$\delta^{left}_{ini}$":"vline",
                             "$\delta^{right}_{ini}$":"vline",
                             "$\delta^{left}_{dsp}$":"vline",
                             "$\delta^{right}_{dsp}$":"vline",
                             "$\delta^{left}_{diff-dsp}$":"vline",
                             "$\delta^{right}_{diff-dsp}$":"vline"}],
                ["(b) Differenced Total Error","c","TE(k)-TE(k-1)",0,8,3,8,
                            {"$TE_{c}-TE_{c-1}$":"curve",
                             "$\delta^{left}_{ini}$":"vline",
                             "$\delta^{right}_{ini}$":"vline",
                             "$\delta^{left}_{diff-dsp}$":"vline",
                             "$\delta^{right}_{diff-dsp}$":"vline"}],
                ["(c) Total Error & Triangle","c","TE & PT",3,8,3,8,
                            {"$PT_{c}$":"curve",
                             "$TE_{c}$":"curve",
                             "$\delta^{left}_{ini}$":"vline",
                             "$\delta^{right}_{ini}$":"vline",}],
                ["(d) Total Error - Triangle","c","TE",6,8,3,8,
                            {"$TE_{c} - PT_{c}$":"curve",
                             "$\delta^{left}_{ini}$":"vline",
                             "$\delta^{right}_{ini}$":"vline",
                             "$\delta^{left}_{dsp}$":"vline",
                             "$\delta^{right}_{dsp}$":"vline"}],
                ["(d) Total Error","c","TE",6,0,3,8,
                            {"$TE_{c}$":"curve",
                             "$\delta^{left}_{ini}$":"vline",
                             "$\delta^{right}_{ini}$":"vline",
                             "$\delta^{left}_{dsp}$":"vline",
                             "$\delta^{right}_{dsp}$":"vline",
                             "$\delta^{left}_{diff-dsp}$":"vline",
                             "$\delta^{right}_{diff-dsp}$":"vline"}],
                             ]
                                 
    def initialize(self):    
        pass

    def transform(self,frame):
        pass_data = frame.getLastPassData()

        img = pass_data[0]         
        te = pass_data[1]
        l = pass_data[2]
        r = pass_data[3]
        
        dt = np.diff(te)
        dt_max = np.argmax(dt)
        dt_min = np.argmin(dt)
        
        li,ri,d,s = triangle(te,left=l,right=r)
            
        rng = np.arange(img.shape[1])

        view_data = {"$I$":img,
                     "$TE_{c}-TE_{c-1}$":[rng[1:],dt],
                     "$TE_{c} - PT_{c}$":[rng,d],
                     "$PT_{c}$":[rng,s],
                     "$TE_{c}$":[rng,te],
                     "$\delta^{left}_{ini}$":l,
                     "$\delta^{right}_{ini}$":r,
                     "$\delta^{left}_{dsp}$":li,
                     "$\delta^{right}_{dsp}$":ri,
                     "$\delta^{left}_{diff-dsp}$":dt_max,
                     "$\delta^{right}_{diff-dsp}$":dt_min
                     }
                     
        pass_data = [img,li,ri]
        
        frame.addData(self.getName(),pass_data,view_data)
        
class Threshold(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Threshold"                                  
    __VIEW__ = [["(a) Gray-scale","c","r",0,0,6,8,
                            {"$I^{gray}$":"image",
                             "$\delta^{left}_{ini}$":"vline",
                             "$\delta^{right}_{ini}$":"vline",}],
                ["(b) Mean Row Intensity","r","M",0,8,6,8,
                            {"$M_{r0}$":"curve",
                             "$PT_{r0}$":"curve"}],
                ["(c) Triangle Algorithm","c","r",6,0,6,8,
                            {"$I^{gray} \geq t_{optimal}$":"image",
                             "$\delta^{left}_{ini}$":"vline",
                             "$\delta^{right}_{ini}$":"vline",}],
                ["(d) Mean Row Intensity","r","M",6,8,6,8,
                            {"$M_{rt^{opt}}$":"curve",
                             "$PT_{rt^{opt}}$":"curve"}],
                ["(e) Otsu's Method","c","r",12,0,6,8,
                            {"$I^{gray} \geq t_{otsu}$":"image",
                             "$\delta^{left}_{ini}$":"vline",
                             "$\delta^{right}_{ini}$":"vline",}],
                ["(f) Mean Row Intensity","r","M",12,8,6,8,
                            {"$M_{rt^{otsu}}$":"curve",
                             "$PT_{rt^{otsu}}$":"curve"}],
                ["(g) Cost Function","t","J",18,0,6,16,
                            {"$J_{t}$":"curve",
                             "$t^{opt}$":"vline",
                             "$t^{otsu}$":"vline"}],
                             ]
                                 
    def initialize(self):    
        pass

    def transform(self,frame):
        pass_data = frame.getLastPassData()
        img = pass_data[0]
        d1n = pass_data[1]
        d2n = pass_data[2]
      
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        otsu = np.copy(gray)
        triangle = np.copy(gray)
 
        l = gray[:,:d1n]
        r = gray[:,d2n:]
        lr = np.hstack((l,r))

        m_gray = np.mean(lr,axis=1)      
        li,ri,d_gray,s_gray = triangle(m_gray,left=0,right=len(m_gray)-1)       

        t_otsu, lr_otsu = cv2.threshold(lr,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)

        t_rng,t = threshold(lr,0,255,below=0)           
        t_triangle = t_rng[np.argmin(t)]

        otsu[:,:d1n] = lr_otsu[:,:d1n]
        otsu[:,d2n:] = lr_otsu[:,d1n:]        
        
        tl = triangle[:,:d1n]
        tl[tl < t_triangle] = 0
        tr = triangle[:,d2n:]
        tr[tr < t_triangle] = 0
        total = np.hstack((tl,tr))
        m_triangle = np.mean(total,axis=1)
        
        tl = otsu[:,:d1n]
        tl[tl < t_otsu] = 0
        tr = otsu[:,d2n:]
        tr[tr < t_otsu] = 0
        total = np.hstack((tl,tr))
        m_otsu = np.mean(total,axis=1)
           
        li,ri,d_triangle,s_triangle = triangle(m_triangle,left=0,right=len(m_gray)-1) 
        lio,rio,d_otsu,s_otsu = triangle(m_otsu,left=0,right=len(m_gray)-1) 
               
        mi = np.argmax(m_triangle)
        
        rng = np.arange(img.shape[0])

        pass_img = img[li:mi,:,:]
        
        view_data = {"$I^{gray}$":gray,
                     "$I^{gray} \geq t_{optimal}$":triangle,
                     "$I^{gray} \geq t_{otsu}$":otsu,
                     "$M_{r0}$":[rng,m_gray],
                     "$M_{rt^{opt}}$":[rng,m_triangle],
                     "$M_{rt^{otsu}}$":[rng,m_otsu],
                     "$PT_{r0}$":[rng,s_gray],
                     "$PT_{rt^{opt}}$":[rng,s_triangle],
                     "$PT_{rt^{otsu}}$":[rng,s_otsu],
                     "$\delta^{left}_{ini}$":d1n,
                     "$\delta^{right}_{ini}$":d2n,
                     "$J_{t}$":[t_rng,t],
                     "$t^{opt}$":t_triangle,
                     "$t^{otsu}$":t_otsu
                     }
                  
        pass_data = [pass_img,d1n,d2n]
        
        frame.addData(self.getName(),pass_data,view_data)
        
class SegmentDeposition(Transformation):
    
    __INST__ = 0   
    __NAME__ = "SegmentDeposition"                                  
    __VIEW__ = [["RGB","$c$","$r$",0,0,3,4,
                            {"$I$":"image",
                             "$\delta_1$":"vline",
                             "$\delta_2$":"vline"}],
                ["TOTAL","$c$","$v$",0,4,3,4,
                            {"total":"curve",
                             "slope":"curve",
                             "$\delta_1$":"vline",
                             "$\delta_2$":"vline"}],
                             ]
                              
    def initialize(self):    
        self.delta = 130

    def transform(self,frame):
        pass_data = frame.getLastPassData()


        img = pass_data[0]     

        
        d1 = self.delta
        d2 = img.shape[1] - self.delta
    
        #ex_list = F.clustering(img,left=d1,right=d2)

        dn1,dn2,t,s = findDeposition(img,left=d1,right=d2)
 
        view_data = {"$I$":img,
                     "$\delta_1$":dn1,
                     "$\delta_2$":dn2,
                     "total":[np.arange(len(t)),t],
                     "slope":[np.arange(len(t)),s]
                              }
                     
        pass_data = [img,dn1,dn2]
        
        frame.addData(self.getName(),pass_data,view_data)
        
class SegmentLineEdge(Transformation):
    
    __INST__ = 0   
    __NAME__ = "SegmentLineEdge"                                  
    __VIEW__ = [["Gray-scale","$c$","$r$",0,0,6,8,
                            {"$I^{gray} > t_{optimal}$":"image",
                             "$\delta_1$":"vline",
                             "$\delta_2$":"vline",
                             "$\delta_3$":"hline",
                             "$\delta_4$":"hline",
                             "$\delta_5$":"hline"}],
                ["Row Mean","$r$","",6,0,3,8,
                            {"$M_{rt_{optimal}}$":"curve",
                             "$\delta_3$":"vline",
                             "$\delta_4$":"vline",
                             "$\delta_5$":"vline"}],
                             ]
                              
    def initialize(self):    
        self.last_t = None
        self.delta = 5
        self.lt = 0
        self.rt = 255
        
    def transform(self,frame):
        pass_data = frame.getLastPassData()

        img = pass_data[0]     
        dn1 = pass_data[1] 
        dn2 = pass_data[2] 
  
        g = cv2.cvtColor(img ,cv2.COLOR_RGB2GRAY)  
        d3,d5,d4,row,topt,rng,t = findLineEdge(g,dn1,dn2,self.lt,self.rt,side="out")   

        self.lt = max((topt-self.delta,0))
        self.rt = min((topt+self.delta,255))  
        
        view_data = {"$I^{gray} > t_{optimal}$":g,
                     "$\delta_1$":dn1,
                     "$\delta_2$":dn2,
                     "$\delta_3$":d3,
                     "$\delta_4$":d4,
                     "$M_{rt_{optimal}}$":[np.arange(len(row)),row],
                     "$\delta_5$":d5,
                              }
                     
        pass_data = [img,dn1,dn2,d3,d4,d5,topt]
        
        frame.addData(self.getName(),pass_data,view_data)
        
class ExtractWidth(Transformation):
    
    __INST__ = 0   
    __NAME__ = "ExtractWidth"                                  
    __VIEW__ = [["RGB","$c$","$r$",0,0,6,8,
                            {"$I^{rgb}$":"image",
                             "$\delta^{NEW}_1$":"vline",
                             "$\delta^{NEW}_2$":"vline",
                             "$\delta_3$":"hline",
                             "$\delta_4$":"hline",
                             "$\delta_6$":"vline",
                             "$\delta_7$":"vline"}],
                ["Total Error","$c$","",6,0,3,8,
                            {"$TE_c$":"curve",
                             "$\delta^{NEW}_1$":"vline",
                             "$\delta^{NEW}_2$":"vline",
                             "$\delta_6$":"vline",
                             "$\delta_7$":"vline"}],
                             ]
                              
    def initialize(self):    
        pass

    def transform(self,frame):
        pass_data = frame.getLastPassData()


        img = pass_data[0]     
        dn1 = pass_data[1]
        dn2 = pass_data[2]
        d3 = pass_data[3]
        d4 = pass_data[4]
        d5 = pass_data[5]
        t_bed = pass_data[6]
        
        edge = img[d3:d4,:,:]
    
        d6,d7,t,s = findDeposition(edge,left=dn1,right=dn2)

        view_data = {"$I^{rgb}$":img,
                     "$\delta^{NEW}_1$":dn1,
                     "$\delta^{NEW}_2$":dn2,
                     "$\delta_3$":d3,
                     "$\delta_4$":d4,
                     "$\delta_6$":d6,
                     "$\delta_7$":d7,
                     "$TE_c$":[np.arange(len(t)),t],
                              }
                     
        pass_data = [img,dn1,dn2,d3,d4,d5,t_bed,d6,d7]
        
        frame.addData(self.getName(),pass_data,view_data)
        
class ExtractHeight(Transformation):
    
    __INST__ = 0   
    __NAME__ = "ExtractHeight"                                  
    __VIEW__ = [["Gray-scale","$c$","$r$",0,0,6,8,
                            {"$I^{gray} > t_{optimal}$":"image",
                             "$\delta_8$":"hline",
                             "$\delta_9$":"hline"}],
                ["Row Mean","$r$","",6,0,3,8,
                            {"$M_{rt_{optimal}}$":"curve",
                             "$\delta_8$":"vline",
                             "$\delta_9$":"vline",
                             "$\delta_{10}$":"vline"}],
                             ]
                              
    def initialize(self):    

        self.delta = 5
        
        self.lt = 0
        self.rt = 255    
        
    def transform(self,frame):
        pass_data = frame.getLastPassData()

        img = pass_data[0]     
        d5 = pass_data[5]
        d6 = pass_data[7]
        d7 = pass_data[8]

        g = cv2.cvtColor(img ,cv2.COLOR_RGB2GRAY)  
        d8,d10,d9,row,topt,rng,t = findLineEdge(g,d6,d7,self.lt,self.rt,
                                            side="in",below=d5)   
        
        self.lt = max((topt-self.delta,0))
        self.rt = min((topt+self.delta,255))      

        view_data = {"$I^{gray} > t_{optimal}$":g,
                     "$\delta_8$":d8,
                     "$\delta_9$":d9,
                     "$M_{rt_{optimal}}$":[np.arange(len(row)),row],
                     "$\delta_{10}$":d10,
                              }
                     
        pass_data = [d10-d5,d7-d6]
     
        frame.addData(self.getName(),pass_data,view_data)
        
class WidthHeight(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Width & Height"                                  
    __VIEW__ = [["Width","frame","pixels",0,0,3,8,
                            {"h":"point"}],
                ["Height","frame","pixels",3,0,3,8,
                            {"w":"point"}],
                             ]
               
    def initialize(self):        
        pass
            
    def transform(self,frame):
        pass_data = frame.getLastPassData()

        h = pass_data[0]     
        w = pass_data[1]
        
            
        view_data = {"h":h,"w":w}
                     
        pass_data = [w,h]
        
        frame.addData(self.getName(),pass_data,view_data)
    
class Complete(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Complete"                                  
    __VIEW__ = [["(a) Rotated Image","c","r",0,0,6,8,
                            {"I":"image",
                             "left_ini":"vline",
                             "right_ini":"vline",
                             "left_dep":"vline",
                             "right_dep":"vline",
                             "left_width":"vline",
                             "right_width":"vline"}],
                ["(b) Triangle Thresholded","c","r",0,8,6,8,
                            {"GT":"image",
                             "upper_bed":"hline",
                             "edge_bed":"hline",
                             "lower_bed":"hline",
                             "upper_dep":"hline",
                             "edge_dep":"hline",
                             "lower_dep":"hline",}],
                ["(c) Deposition Segmentation","c","TE",6,0,3,8,
                            {"TE_bed":"curve",
                             "left_ini":"vline",
                             "right_ini":"vline",
                             "left_dep":"vline",
                             "right_dep":"vline"}],
                ["(d) Bed Edge Segmentation","r","M",6,8,3,8,
                            {"M_bed":"curve",
                             "upper_bed":"vline",
                             "edge_bed":"vline",
                             "lower_bed":"vline"}],
                ["(e) Width Extraction","c","TE",9,0,3,8,
                            {"TE_dep":"curve",
                             "left_dep":"vline",
                             "right_dep":"vline",
                             "left_width":"vline",
                             "right_width":"vline"}],
                ["(f) Deposition Edge Segmentation","r","M",9,8,3,8,
                            {"M_dep":"curve",
                             "upper_dep":"vline",
                             "edge_dep":"vline",
                             "lower_dep":"vline"}],
                            ]
#                ["(g) Real Time Width","frame number","pixels",12,0,3,8,
#                            {"W":"point"}],
#                ["(h) Real Time Height","frame number","pixels",12,8,3,8,
#                            {"H":"point"}]
#                             ]

    __SETS__ = [["Initial",[["Size","L",0,0,1,1],
                            ["S1","S",0,1,1,1,130,0,1000],
                           ]  
                ],
                ["Thresholding",[["Bed","L",0,0,1,1],
                                 ["bed","S",0,1,1,1,123,0,255],
                                 ["Dep","L",1,0,1,1],
                                 ["dep","S",1,1,1,1,247,0,255],
                                 ["Delta","L",2,0,1,1],
                                 ["delta","S",2,1,1,1,10,0,255],
                                ]
                ]
               ]
                
    def initialize(self):    
        self.delta = 130
        self.bed_t = 200
        self.dep_t = 247
        self.delta_t = 10
        self.bed_min_t = self.bed_t -self.delta_t
        self.bed_max_t = self.bed_t +self.delta_t
        self.dep_min_t = self.dep_t -self.delta_t
        self.dep_max_t = self.dep_t +self.delta_t
        
    def transform(self,frame):
        device_data = frame.getLastPassData()
        
        I = device_data[0]  
        G = cv2.cvtColor(I ,cv2.COLOR_RGB2GRAY)  
        
        rows = np.array(range(I.shape[0]) )
        cols = np.array(range(I.shape[1]) )
        
        left_ini = self.delta
        right_ini = I.shape[1] - self.delta
        
        one = findDeposition(I,left=left_ini,right=right_ini)
        
        if verify(one[2],one[0],one[1],left_ini,right_ini):
            
            two = findLineEdge(G,one[0],one[1],
                                 self.bed_min_t,self.bed_max_t)  
            
            I_bed = I[two[0]:two[2],:,:]
    
            three = findDeposition(I_bed,left=one[0],right=one[1])  
            
            if verify(three[2],three[0],three[1],one[0],one[1]):
                four = findLineEdge(G,three[0],three[1],
                                      self.dep_min_t,self.dep_max_t,
                                      side="in",below=0)
                
                if four[1] >= two[1]:
                    H = four[1] - two[1]
                else:
                    H = 0
                    
                W = three[1] - three[0]
            else:
                zero_rows = np.zeros(I.shape[0])
                four = [0,0,0,zero_rows,0,rows,zero_rows]
                
                H = 0
                W = 0       
        else:
            zero_cols = np.zeros(I.shape[1])
            zero_rows = np.zeros(I.shape[0])

            two = [0,0,0,zero_rows,0,rows,zero_rows]
            three = [0,0,zero_cols,zero_cols]
            four = two
            
            H = 0
            W = 0
        
        print(two[4],four[4])
        
        view_data = {"I":I,
                     "left_ini":left_ini,
                     "right_ini":right_ini,
                     "TE_bed":[cols,one[2]],
                     "left_dep":one[0],
                     "right_dep":one[1], 
                     "TE_dep":[cols,three[2]],
                     "left_width":three[0],
                     "right_width":three[1],
                     "GT":G,
                     "upper_bed":two[0],
                     "edge_bed":two[1],
                     "lower_bed":two[2],
                     "M_bed":[rows,two[3]],
                     "upper_dep":four[0],
                     "edge_dep":four[1],
                     "lower_dep":four[2],
                     "M_dep":[rows,four[3]],  
#                     "H":H,
#                     "W":W
                     }
                     
        pass_data = [H,W]
        
        frame.addData(self.getName(),pass_data,view_data)
        
    def setParams(self,par_list):
        name = par_list[0]
        value = par_list[1] 
        
        if name == "bed":
            self.bed_t = value
        elif name == "dep":
            self.dep_t = value
        elif name == "delta":
            self.delta_t = value


class CompleteXiris3(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Complete-Xiris-3"                                  
    __VIEW__ = [["Rotated and Thresholded","c","r",0,0,6,8,
                            {"I":"image",
                             "upper_bed":"hline",
                             "edge_bed":"hline",
                             "lower_bed":"hline",
                             "upper_dep":"hline",
                             "edge_dep":"hline",
                             "lower_dep":"hline",}],
                ["Edge Segmentation","r","M",6,0,3,8,
                            {"M_bed":"curve",
                             "M_dep":"curve",
                             "upper_bed":"vline",
                             "edge_bed":"vline",
                             "lower_bed":"vline",
                             "upper_dep":"vline",
                             "edge_dep":"vline",
                             "lower_dep":"vline",}]
                            ]
                
    def initialize(self):    
        self.delta = 130
        
    def findEdge(self,G,left,right,side='out'):
        
        if side == 'out':
            side = np.hstack((G[:,:left],G[:,right:]))
        elif side == 'in':
            side = G[:,left:right]
            
        sm = np.sum(side,axis=1)    
        li,ri,d,s = triangle(sm, 0, len(sm)-1)
                               
        mi = np.argmax(sm)
        mid = midval(sm,li,mi)
        
        return (li,mid,mi,sm)

    def findDeposition(self, ch, left, right ):     
    
        sse = template(ch,left=left,right=right)
        total = normalize(sse,left=left,right=right)    
        li,ri,d,s = triangle(total,left=left,right=right)
        
        return (li,ri,total)
        
    def transform(self,frame):
        device_data = frame.getLastPassData()
        
        G = device_data[0]  
        
        rows = np.array(range(G.shape[0]) )
        cols = np.array(range(G.shape[1]) )
        
        left_ini = self.delta
        right_ini = G.shape[1] - self.delta
        
        one = self.findDeposition(G,left_ini, right_ini)   
            
        two = self.findEdge(G,one[0],one[1],side='out')      
        four = self.findEdge(G,one[0],one[1],side='in')  
            
        if four[1] >= two[1]:
            H = four[1] - two[1]
        else:
            H = 0
            
        W = one[1] - one[0]
        
        view_data = {"I":G,                        
                     "upper_bed":two[0],
                     "edge_bed":two[1],
                     "lower_bed":two[2],
                     "upper_dep":four[0],
                     "edge_dep":four[1],
                     "lower_dep":four[2], 
                     "M_dep":[rows,four[3]],  
                     "M_bed":[rows,two[3]],  
                     }
                     
        pass_data = [H,W]
        
        frame.addData(self.getName(),pass_data,view_data)
        
class CompleteXiris2(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Complete-Xiris-2"                                  
    __VIEW__ = [["Rotated and Thresholded","c","r",0,0,6,8,
                            {"I":"image",
                             "left_ini":"vline",
                             "right_ini":"vline",
                             "left_dep":"vline",
                             "right_dep":"vline"}],
                ["Deposition Segmentation","c","error",6,0,3,8,
                            {"TE_bed":"curve",
                             "left_ini":"vline",
                             "right_ini":"vline",
                             "left_dep":"vline",
                             "right_dep":"vline"}],
                            ]
                
    def initialize(self):    
        self.delta = 130
        
    def findEdge(self,G,left,right,side='out'):
        
        if side == 'out':
            side = np.hstack((G[:,:left],G[:,right:]))
        elif side == 'in':
            side = G[:,left:right]
            
        sm = np.sum(side,axis=1)    
        li,ri,d,s = triangle(sm, 0, len(sm)-1)
                               
        mi = np.argmax(sm)
        mid = midval(sm,li,mi)
        
        return (li,mid,mi,sm)

    def findDeposition(self, ch, left, right ):     
    
        sse = template(ch,left=left,right=right)
        total = normalize(sse,left=left,right=right)    
        li,ri,d,s = triangle(total,left=left,right=right)
        
        return (li,ri,total)
        
    def transform(self,frame):
        device_data = frame.getLastPassData()
        
        G = device_data[0]  
        
        rows = np.array(range(G.shape[0]) )
        cols = np.array(range(G.shape[1]) )
        
        left_ini = self.delta
        right_ini = G.shape[1] - self.delta
        
        one = self.findDeposition(G,left_ini, right_ini)   
            
        two = self.findEdge(G,one[0],one[1],side='out')      
        four = self.findEdge(G,one[0],one[1],side='in')  
            
        if four[1] >= two[1]:
            H = four[1] - two[1]
        else:
            H = 0
            
        W = one[1] - one[0]
        
        view_data = {"I":G,
                     "left_ini":left_ini,
                     "right_ini":right_ini,
                     "left_dep":one[0],
                     "right_dep":one[1],                           
                     "upper_bed":two[0],
                     "edge_bed":two[1],
                     "lower_bed":two[2],
                     "upper_dep":four[0],
                     "edge_dep":four[1],
                     "lower_dep":four[2], 
                     "TE_bed":[cols,one[2]],
                     "M_dep":[rows,four[3]],  
                     "M_bed":[rows,two[3]],  
                     }
                     
        pass_data = [G]
        
        frame.addData(self.getName(),pass_data,view_data)
        
class CompleteXiris(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Complete-Xiris"                                  
    __VIEW__ = [["Column-wise Thresholded and Rotated","c","r",0,0,6,8,
                            {"GT":"image"}],
                ["Thresholds","column","value",6,0,3,8,
                            {"t":"curve"}],
                            ]

    __SETS__ = [["Rotation",[["Angle","L",0,0,1,1],
                             ["angle","S",0,1,1,1,-170,-450,450],
                                ]
                ]
               ]
                
    def initialize(self):    
        self.angle = -170
        self.delta = 130
          
    def transform(self,frame):
        device_data = frame.getLastPassData()
        
        G = device_data[0]   
        S = np.sort(G, axis=0)
        
        sm = np.sum(S,axis=1)
        arg = np.argmax(np.diff(sm))
        
        ts = S[arg,:]

        G[G < ts] = 0
        
        left_ini = self.delta
        right_ini = G.shape[1] - self.delta
        
        M = cv2.getRotationMatrix2D( (G.shape[1]/2, G.shape[0]/2) , float(self.angle)/100,1)
        rotated = cv2.warpAffine(G,M,(G.shape[1],G.shape[0]) )
        
        lc = rotated[ :, :left_ini ]
        rc = rotated[ :, right_ini: ]      
        
        slc = np.sum(lc, axis=1)
        src = np.sum(rc, axis=1)    

        
        rows = np.array(range(G.shape[0]) )
        cols = np.array(range(G.shape[1]) )

        
        view_data = {"t":[cols,ts],
                     "GT":rotated,
                     }
                     
        
        pass_data = [rotated]
        
        frame.addData(self.getName(),pass_data,view_data)
    
    def setParams(self,par_list):
        name = par_list[0]
        value = par_list[1] 
        
        if name == "angle":
            self.angle = value

class MeltpoolThreshold(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Meltpool Threshold"                                  
    __VIEW__ = [["Thresholded","c","r",0,0,3,8,
                            {"I":"image"}],
                ["Sorted","c","r",0,8,6,8,
                            {"G":"image"}],
                ["babh","c","r",3,0,3,8,
                            {"c":"curve"}],
                            ]

    __SETS__ = [["Contour D",[["Threshold","L",0,0,1,1],
                             ["t","S",0,1,1,1,0,0,4096],
                             ["Contours","L",1,0,1,1],
                             ["y","S",1,1,1,1,0,0,2000],
                                ]
                ]
               ]
                
    def initialize(self):    
        self.amount = 0
        self.S = []
        self.x = 0
        self.y = 0
        self.t = 0
        
    def left_triangle(self,arr):
        slope = np.linspace(arr[0],arr[-1],num=len(arr), endpoint=True) 
        diff = arr-slope
        
        point = np.argmin(diff)
        return point

    def right_triangle(self,arr):
        arr = arr.flatten()
        slope = np.copy(arr)
        slope[np.argmax(arr):] = np.linspace(max(arr),arr[-1],num=len(arr)-np.argmax(arr), endpoint=True) 
        diff = arr-slope
        
        point = np.argmin(diff)
        return point,slope
    
    def contour_check(self,region, contour,shape):

        mask = np.zeros(shape)
        # check if contour fits
                
        # if the area is bigger than it would never fit
        if cv2.contourArea(region) <= cv2.contourArea(contour):
            # set all pixels of the samller region to zero
            cv2.fillPoly(mask,region,255)
            cv2.fillPoly(mask,contour,0)
            
            if np.any(mask):
                return False
            else:
                return True
        else:
            return False

    def make_dams(self,insides,regions,contour,total):
                    # create mask


        mask = np.zeros(total.shape)      

        kernel = np.ones((3,3),np.uint8)
        i = 0;
    
        cv2.fillPoly(mask,contour,255)
        while cv2.countNonZero(mask) == 0:
            print("Make Dams Iteration: ",i)

            for index,x in enumerate(insides):
                temp = np.zeros(total.shape)
                cv2.fillPoly(temp,regions[index],255)
                dilated = cv2.dilate(temp,kernel,iterations=i)
                
                cv2.bitwise_and(mask,dilated,total)
                
            i+=1
            
        return total

    def sorted_contours(self,im,t):
        
        c = np.copy(im)
        c = self.scale(c)
        c[c<t] = 0
        i,contours,h = cv2.findContours(c, 
                                 cv2.RETR_TREE,
                                 cv2.CHAIN_APPROX_NONE)
        
        s = sorted(contours,key=lambda x: cv2.contourArea(x))
        s.reverse()
        return c,s
       
    def scale(self,img):
        mn = np.amin(img)
        mx = np.amax(img)
        
        df = mx-mn
        
        power = math.log(df,2)
        
        if power <= 8:
            factor = 2**8-1
            tp = np.uint8
        elif power > 8 and power <= 16:
            factor =  2**16-1
            tp = np.uint16
            
        im = (img - mn ) / (mx - mn )*factor
        return im.astype(tp)        
              
    def watershed(self,arr):

        im = self.scale(arr)
        
        s = im.shape
        regions = []

        mask = np.zeros(s)
        for t in np.arange(255,230,-1): # for every threshold

            cs = self.sorted_contours(im,t)
            print("Regions: ",len(regions),
                  " Threshold: ",t, 
                  " Nr Contours: ",len(cs))
            
            for c in cs: # contours
                # check if existing contours fall into new contour            
                # create mask

                insides = [self.contour_check(r,c,s) for r in regions] 
                # count the number of fits
                nr = insides.count(True)
                # if no region fits in the new contour than a new region
                # is found
                if nr == 0:
                    print("new")
                    regions.append(c)
                    continue
                # if none of the existing contours falls into the new one
                # then add that contour to the list as a new region
                elif nr == 1:
                    print("update: ",insides.index(True))
                    regions[insides.index(True)] = c
                    continue
                # If multiple contours fall in the new contour then
                # we have a merge and we have to find the DAMS by dilation
                # of the existing contours until they overlap.
                else:
                    print("multiple")
                    # get indexes of regions which fit
                    
                    total = np.zeros(im.shape)
                    return self.make_dams(insides,regions,c,total)

        
        #return regions
      
    def dotriangle(self,rowcol):

        l,r,d,s = triangle(rowcol,0,len(rowcol)-1)
        out = np.copy(rowcol)
        if verify(rowcol,l,r,0,len(rowcol)-1):
            out[:l] = 0
            out[r:] = 0
        else:
            out[:] = 0
        
        return out
    
    def upper_hull(self,arr):
        
        x = np.arange(len(arr)).astype(float)
        y = arr.astype(float)
        points = list(zip(x,y))
        
        upper = []
        for p in points:
            while len(upper) >= 2 and self.CCW(upper[-2],upper[-1],p):
                upper.pop()
            upper.append(p)
            
        p = np.array(upper,dtype=int)
        a = np.interp(x.astype(int),p[:,0],p[:,1])
        return a,p
        
    def lower_hull(self,arr):
        mn = min(arr)
        mx = max(arr)
        m = (mx-mn) + mn
        flipped = -(arr-m) + m
        
        h,points = self.upper_hull(flipped)
        lh = -(h - m) + m
        return lh,points
    
    def flipped(self,arr):
        mn = min(arr)
        mx = max(arr)
        m = (mx-mn) + mn
        flipped = -(arr-m) + m  
        return flipped
        
    def CCW(self,p1, p2, p3):
    	if (p3[1]-p1[1])*(p2[0]-p1[0]) >= (p2[1]-p1[1])*(p3[0]-p1[0]):
    		return True
    	return False
        
    def get_edge_points(self,point,image,center):
        arr,x,y = self.get_array(image,point,center)
        pts,pos,lh,df = self.get_positions(arr)
        return [(int(x[i[0]]),int(y[i[0]]),arr[i[0]]) for i in pts if i[0] < pos]
                
        
        
        
    def max_spin_round(self,image):
        center = self.max_center(image)
        points = self.get_points(image,center)
        
        out = np.zeros(image.shape).astype(np.uint16)
        out2 = np.copy(out)
        for p in points:
            arr,x,y = self.get_array(image,p,center)
            pts,pos,lh,df = self.get_positions(arr)
            v = np.array([ i[0] for i in pts if i[0]<pos])
            
            absdf = abs(df) / float(max(df)) * 255
            out[x[v],y[v]] = absdf[v]
            aw = np.zeros(arr.shape)
            aw[pos] = 255
            out2[x,y] = aw

            #valid_points += [(int(x[i[0]]),int(y[i[0]]),arr[i[0]]) for i in pts if i[0] < pos]
        out = out / float(np.amax(out)) * 255
        return out.astype(np.uint8),out2

    def lower_image_hull(self,image):
        c = self.max_center(image)
        points = self.get_points(image,c)

        out = np.zeros(image.shape) 
        out2 = np.copy(out).astype(np.uint8)
        for i,p in enumerate(points):
            l,x1,y1 = self.get_array(image,points[i-1],c)  
            m,x2,y2 = self.get_array(image,points[i],c)  
            r,x3,y3 = self.get_array(image,points[i+1],c)  
            
            lng = np.min((len(l),len(m),len(r)))
            
            arr = np.min((l[-lng:],m[-lng:],r[-lng:]),axis=1)
            lh,points = self.lower_hull(arr)
            
            #out2[points[:,1],points[:,0]] = 255
            out[x1,y1] = lh
            out[x2,y2] = lh
            out[x3,y3] = lh
            
        return out,out2
      
    def getR(self,image):
        c = self.max_center(image)
        points = self.get_points(image,c)
        print(len(points),2*sum(image.shape))
        
        s = image.shape
        l = np.max((s[0],s[1]))
        
        out = np.zeros((l,2*sum(s))) 
        
        for i in range(len(set(points))):
            arr,x1,y1 = self.get_array(image,points[i],c)  
            l = len(arr)
            out[:l,i] = arr
            
        return out
    
                
        
    def max_center(self,image):
        ind = np.unravel_index(np.argmax(image,axis=None),
                               image.shape)
        y0 = ind[0]
        x0 = ind[1]
        
        return (y0,x0)
    
    def get_points(self,image,center):
        
        s = image.shape

        out = []
        for c,v in np.ndenumerate(image):
            row = c[0]
            col = c[1]
            if col == 0 or col == s[1]-1:
                out.append(c)
            elif row == 0 or row == s[0]-1:
                out.append(c)
        
        return out
    
    def get_array(self,image,point,center):
        
        length = int(np.sqrt((center[0]-point[0])**2+(center[1]-point[1])**2))

        x = np.linspace(center[0],point[0],length).astype(np.int)
        y = np.linspace(center[1],point[1],length).astype(np.int)
        
        z = image[x.astype(np.int),y.astype(np.int)]        
        
        return z,x,y
    
    
    def get_positions(self,arr):

        lh,pts = self.lower_hull(arr)
        triangle = np.linspace(lh[0],lh[-1],num=len(arr))     
        diff = triangle - lh
        pos = np.argmax(diff)
        
        return pts,pos,lh,diff
        
    def draw_points(self,shape, points):
        out = np.zeros(shape).astype(np.uint16)
        for p in points:
            out[p[0],p[1]] = p[2]
            
        return out
        
    def staircase(self,arr):
        
        out = np.zeros(arr.shape)
        out[0] = arr[0]
        for i in np.arange(1,len(arr)-1):
            if arr[i] < out[i-1]:    
                out[i] = arr[i]
            else:
                out[i] = out[i-1]
                
        return out
        
    def transform(self,frame):
        device_data = frame.getLastPassData()
        
        G = device_data[0]   
        G = G - np.amin(G,axis=None)

        mn = np.amin(G)
        mx = np.amax(G)
        
        factor = 2**8-1
        tp = np.uint8
            
        im = (G - mn ) / (mx - mn )*factor
        im = im.astype(tp)
        
        Gc = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
                
        """
        vp = self.max_spin_round(im)
        
        vals = np.array([i[2] for i in vp])
        
        mx = np.max(vals)
        mn = np.min(vals)
        binc = mx-mn
        
        hist,bins = np.histogram(vals.ravel(),int(binc),[mn,mx])
        
        out = self.draw_points(G.shape,vp)
        im2,cs = self.sorted_contours(G,self.t)

        cs.reverse()

        im2,cs = self.sorted_contours(G,self.t)
        cs.reverse()
        print(cs[0])
        gc = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
      
        
        
        cv2.drawContours(gc,cs,0,(0,255,0),3)
        cv2.ellipse(gc,ellipse,(255,0,0),2)
            

        
        """  
        
        #Gh,Gp = self.max_spin_round(G)
        #lh,s = self.lower_image_hull(G)
        R = self.getR(G)
        #Gh[Gh < self.t] = 0
        #Gh[Gh > self.y] = 0
        #pos = np.nonzero(Gh)
        
        #hist = cv2.calcHist([Gh[pos]],[0],None,[256],[0,256])
        #h = hist.flatten().astype(int)
        rng = np.arange(10)        

        #x = pos[0]
        #y = pos[1]
        #cont = np.array(list(set(zip(y,x))))
        #ellipse = cv2.fitEllipse(cont)
        #cv2.ellipse(im,ellipse,(0,255,0),2)
        #cv2.ellipse(Gh,ellipse,(127),2)




        view_data = {"I":im,
                     "G":R,
                     "c":[rng,rng]
                     }
                     
        
        pass_data = [G]
        
        frame.addData(self.getName(),pass_data,view_data)
    
    def setParams(self,par_list):
        name = par_list[0]
        value = par_list[1] 
        
        if name == "t":
            self.t = value
        elif name == "y":
            self.y = value
        
class SortThreshold(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Sort Threshold"                                  
    __VIEW__ = [["Thresholded","c","r",0,0,6,8,
                            {"I":"image",
                              "top":"hline",
                              "bottom":"hline",
                              "m":"curve"}],
                ["Sorted","c","r",0,8,6,8,
                            {"S":"image"}],
                ["Thresholds","column","value",6,0,3,8,
                            {"t":"curve",
                             "t2":"curve"}],
                ["Sort Mean","column","value",6,8,3,8,
                            {"s_mean":"curve"}]
                            ]
                
    __SETS__ = [["ROI",[["Top","L",0,0,1,1],
                        ["Down","L",1,0,1,1],
                        ["top","S",0,1,1,1,0,0,5000],
                        ["down","S",1,1,1,1,0,0,5000],
                        ]
                        ],
                ["Channel",[["Select","L",0,0,1,1],
                        ["select","S",0,1,1,1,0,0,3],
                        ["Row","L",1,0,1,1],
                        ["row","S",1,1,1,1,0,0,5000],
                        ]
                        ]
                ]
                    
    def initialize(self):    
        self.top = 0
        self.down = 0
        self.select = 0
        self.row = 0

    def setZero(self,I,li,ri,i):
        I[:li,i] = 0
        I[ri:,i] = 0
       
    def gaussian(self,x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)

    def balance(self,image):
        mids = []
        for col in np.arange(image.shape[1]):
            row = image[:,col].flatten()
            res = minimize_scalar(self.f,args=(row,))
            mids.append(res.x)
            
        return np.array(mids)
            
    def f(self,x,row):     
        x = int(x)
        return np.abs(np.sum(row[:x]) - np.sum(row[x:])) 
    
    def getTemplate3(self,sm1):
        delta = 50
        stk = np.hstack((sm1,sm1[::-1]))
        li1,ri1,d1,s1 = triangle(stk,0,len(stk)-1) 
        template = scipy.signal.resample(stk[li1:ri1],int((ri1-li1)/2)).astype(np.uint8)

        return template,int(li1/2),int(ri1/2)
    
    def getTemplate2(self,sm1):
        odd = sm1[::2]
        even = sm1[1::2]
        tmp_side = (odd+even)/2
        tmp = np.hstack((tmp_side,tmp_side[::-1]))
        li1,ri1,d1,s1 = triangle(tmp,0,len(tmp)-1) 
        template = tmp.astype(np.uint8)
        return template,li1,ri1
        
    def getTemplate(self,sm1):
        stk = np.hstack((sm1,sm1[::-1]))
        tmp = cv2.resize(stk,(len(sm1),1)).flatten()

        li1,ri1,d1,s1 = triangle(tmp,0,len(tmp)-1) 
        print(li1,ri1)
        template = tmp[li1:ri1].astype(np.uint8)
        tmp = np.reshape(template,(1,len(template)))
        return tmp,li1,ri1
    
    def transform(self,frame):
        device_data = frame.getLastPassData()
        
        if self.select < 3:
            G = device_data[0][:,:,self.select].astype(np.uint8)
        else:
            G = cv2.cvtColor(device_data[0],cv2.COLOR_RGB2GRAY)

        s = G.shape

        I = G[self.top:s[0]-self.down,:]
        
        S1 = np.sort(I, axis=0)   

        tmps = [self.getTemplate(S1[:,i]) for i in np.arange(S1.shape[1])]
        print(tmps[0][0].shape)
        ress = [cv2.matchTemplate(I[:,i],tmps[i][0],cv2.TM_CCORR) for i in np.arange(S1.shape[1])]        
        mids = [np.argmax(ress[i])+len(tmps[i][0])/2 for i in np.arange(S1.shape[1])]
        

        rows = np.arange(I.shape[0]).flatten()
        cols = np.arange(I.shape[1]).flatten()
        
        li = tmps[self.row][1]
        ri = tmps[self.row][2]
        col = tmps[self.row][0]
        print(li,ri)
        view_data = {"I":I,
                     "S":I,
                     "m":[cols,mids],
                     "t":[rows,I[:,self.row]],
                     "t2":[rows[li:ri],col],
                     "s_mean":[rows,S1[:,self.row]],
                     "top":self.top,
                     "bottom":s[0]-self.down,
                     }
                     
        
        pass_data = [I]
        
        frame.addData(self.getName(),pass_data,view_data)
    
    def setParams(self,par_list):
        name = par_list[0]
        value = par_list[1] 
        
        if name == "top":
            self.top = value
        elif name == "down":
            self.down = value
        elif name == "select":
            self.select = value
        elif name == "row":
            self.row = value

class HoughLines(Transformation):
    
    __INST__ = 0   
    __NAME__ = "HoughLines"                                  
    __VIEW__ = [["Thresholded","c","r",0,0,6,8,
                            {"I":"image",
                              "top":"hline",
                              "bottom":"hline"}],
                ["Sorted","c","r",0,8,6,8,
                            {"S":"image"}],
                ["Thresholds","column","value",6,0,3,8,
                            {"t":"curve"}],
                ["Sort Mean","column","value",6,8,3,8,
                            {"s_mean":"curve",
                             "s_line":"vline"}]
                            ]
                
    __SETS__ = [["ROI",[["Top","L",0,0,1,1],
                        ["Down","L",1,0,1,1],
                        ["top","S",0,1,1,1,0,0,5000],
                        ["down","S",1,1,1,1,0,0,5000],
                        ]
                        ],
                ["Channel",[["Select","L",0,0,1,1],
                        ["select","S",0,1,1,1,0,0,3],
                        ["Canny Low","L",1,0,1,1],
                        ["cl","S",1,1,1,1,0,0,255],
                        ["Canny High","L",2,0,1,1],
                        ["ch","S",2,1,1,1,0,0,255],
                        ["Hough Low","L",3,0,1,1],
                        ["hl","S",3,1,1,1,0,0,255],
                        ]
                        ]
                ]
                    
    def initialize(self):    
        self.top = 0
        self.down = 0
        self.select = 0
        self.cl = 0
        self.ch = 0
        self.hl = 0

    def addLines(self,edges,img):
        
        
        lines = cv2.HoughLines(edges,1,np.pi/180,self.hl)
        try:

            for line in lines:
                for rho,theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                
                    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        except:
            print("No Lines Found")
    
    def transform(self,frame):
        device_data = frame.getLastPassData()
        
        if self.select < 3:
            G = device_data[0][:,:,self.select].astype(np.uint8)
        else:
            G = cv2.cvtColor(device_data[0],cv2.COLOR_RGB2GRAY)

        out = device_data[0]
        s = G.shape
        I = G[self.top:s[0]-self.down,:]
        S1 = np.sort(I, axis=0)  

        sm1 = np.median(S1,axis=1)
        
        #v = np.diff(sm1)
        li1,ri1,d1,s1 = triangle(sm1,0,len(sm1)-1) 
        v = sm1
        #print(v.shape,sm1.shape)
        #arg = np.argmax(v)  
        arg = li1
        ts = S1[arg,:]

        #I[I < ts] = 0
        rows = np.array(range(len(v)))
        cols = np.array(range(G.shape[1]) )

        edges = cv2.Canny(G,self.cl,self.ch,apertureSize = 3)

        #self.addLines(edges,out)

        view_data = {"I":out,
                     "S":edges,
                     "t":[cols,ts],
                     "s_mean":[rows,v],
                     "top":self.top,
                     "bottom":s[0]-self.down,
                     }
                     
        
        pass_data = [G]
        
        frame.addData(self.getName(),pass_data,view_data)
    
    def setParams(self,par_list):
        name = par_list[0]
        value = par_list[1] 
        
        if name == "top":
            self.top = value
        elif name == "down":
            self.down = value
        elif name == "select":
            self.select = value
        elif name == "cl":
            self.cl = value
        elif name == "ch":
            self.ch = value
        elif name == "hl":
            self.hl = value
            
class TiltCorrection(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Tilt Correction"                                  
    __VIEW__ = [["Column-wise Thresholded and Rotated","c","r",0,0,6,8,
                            {"GT":"image"}],
                ["Side Means","row","mean",6,0,3,8,
                            {"left":"curve",
                             "right":"curve",
                             "left_li":"vline",
                             "right_li":"vline"}],
                            ]
                
    def initialize(self):    
        self.angle = -170
        self.delta = 130
          
    def corr_rotate(self,G, angle):
        s = G.shape

        M = cv2.getRotationMatrix2D( (s[1]/2, s[0]/2) , float(angle)/100,1)
        rotated = cv2.warpAffine(G,M,(s[1],s[0]) )
        
        lc = rotated[ :, :self.delta ]
        rc = rotated[ :, s[1] - self.delta: ]                 

        slc = np.sum(lc, axis=1)
        src = np.sum(rc, axis=1)  
        
        c = np.correlate(slc,src)
        
        return c[0]
    
    def transform(self,frame):
        device_data = frame.getLastPassData()
        
        G = device_data[0]   
        s = G.shape
        #ceffs = np.array([self.corr_rotate(G,i) for i in np.arange(-450,450)]) 

        lc = G[ :, :self.delta ]
        rc = G[ :,s[1] - self.delta: ]                 

        slc = np.sum(lc, axis=1)
        src = np.sum(rc, axis=1)  

        li1,ri1,d1,s1 = triangle(slc,0,s[0]-1) 
        li2,ri2,d2,s2 = triangle(src,0,s[0]-1) 
        

            
        w = s[1]-self.delta

        if li1 < li2:
            h = li2 - li1
            angle = np.arctan(h/w)
        else:
            h = li1 - li2
            angle = -np.arctan(h/w)
        
        print(float(np.rad2deg(angle)))
        M = cv2.getRotationMatrix2D( (s[1]/2, s[0]/2) , float(np.rad2deg(angle)),1)
        rotated = cv2.warpAffine(G,M,(s[1],s[0]) )
        
        rows = np.arange(s[0])
        
        view_data = {"left":[rows,slc],
                     "right":[rows,src],
                     "left_li":li1,
                     "right_li":li2,
                     "GT":rotated,
                     }
                     
        
        pass_data = [rotated]
        
        frame.addData(self.getName(),pass_data,view_data)
    
    def setParams(self,par_list):
        name = par_list[0]
        value = par_list[1] 
        
        if name == "angle":
            self.angle = value
            
class TriangleDirect(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Triangle direct"                                  
    __VIEW__ = [["Rotated and Thresholded","c","r",0,0,6,8,
                            {"I":"image",
                             "mid":"curve",
                             "left":"curve",
                             "right":"curve"}]
                            ]
                
    def initialize(self):    
        pass
             
    def transform(self,frame):
        device_data = frame.getLastPassData()
        
        G = device_data[0]
        s = G.shape
        T2 =  [ (triangle(G[:,i],0,s[0]-1) ) for i in range(s[1]) ]
        mid = [s[0]+(s[1]-s[0])/2 for s in T2]
        left = [s[0] for s in T2]
        right = [s[1] for s in T2]
        
        left = (G!=0).argmax(axis=0)
        right = G.shape[0] - (np.flipud(G)!=0).argmax(axis=0) 
        mid = left+(right - left)/2
        rng = np.arange(len(mid))
        view_data = {"I":G,
                     "mid":[rng,mid],
                     "left":[rng,left],
                     "right":[rng,right],                     
                     }
                     
        pass_data = [G]
        
        frame.addData(self.getName(),pass_data,view_data)
        
class HW(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Height and Width"                                  
    __VIEW__ = [["(g) Real Time Width","frame number","pixels",0,0,3,8,
                            {"W":"point"}],
                ["(h) Real Time Height","frame number","pixels",3,0,3,8,
                            {"H":"point"}]
                             ]
                
    def initialize(self):    
        self.csvfile =  open('hw_1.csv', 'w', newline='') 
        self.writer = csv.writer(self.csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
      
        self.pos = 0
        self.writer.writerow(['Height', 'Width'])
        
    def transform(self,frame):
        hw_data = frame.getLastPassData()
        H = hw_data[0]
        W = hw_data[1]
        pos = frame.getPosition()
        
        if self.pos < pos:
            self.writer.writerow([H, W])
            self.pos = pos
        else:
            print("Saving..")
            self.csvfile.close()
            
        view_data = {"H":H, "W":W}
        pass_data = []
        
        frame.addData(self.getName(),pass_data,view_data)
        
class W_Keyence(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Width Keyence"                                  
    __VIEW__ = [["Height Map","column","row",0,0,6,8,    
                             {"I":"image",
                              "l":"curve",
                              "r":"curve"}],
                ["Row","row","height",6,0,3,8,
                             {"mean":"curve",
                              "std":"curve",
                              "left":"vline",
                              "right":"vline"}],              
                             ]

    __SETS__ = [["Intersection",[["Row","L",0,0,1,1],
                                 ["row2","S",0,1,1,1,0,0,5000],
                                 ["degree","S",1,0,1,1,0,0,3600]
                                 ]
                        ]
                ]
                
    def initialize(self):    
        self.row = 0
        self.degree = 1
        
    def getWidth(self,i,img,left,right):
        col = img[i,:].flatten()
        li,ri,d,s = triangle(col,left=left,right=right)
        
        if verify(col,li,ri,left,right):
            return (li,ri)
        else:
            return (0,0)    
       
    def getHeight(self,i,img):
        col = img[:,i].flatten()  
        height_nanometer = float(max(col) ) /10
        height_mm = height_nanometer / 1000000
        return ( height_mm,np.argmax(col) )
        
    def piecewise_function(self,x,lb,lt,rt,rb,b0,a1,a2,a3):
        cond_list = [x < lb,(x>=lb) & (x<lt),(x>=lt) & (x<=rt),(x>rt) & (x<=rb),x>rb]

        func_list  = [lambda x: b0, 
                   lambda x: b0+ a1*(x-lb), 
                   lambda x: b0+ a1*(x-lb) + a2*(x-rb), 
                   lambda x: b0+ a1*(x-lb) + a2*(x-lt) + a3*(x-rt),
                   lambda x: b0]
        
        return np.piecewise(x,cond_list,func_list)
    
    def transform(self,frame):
        image = frame.getLastPassData()[0]
        
        s = image.shape
        rows = s[0]
        cols = s[1]
        
        row_rng = np.arange(rows)
        col_rng = np.arange(cols)
        
        #lr = [self.getWidth(i,image,left=0,right=cols-1) for i in row_rng]
        lr = [(np.argmax(np.diff(image[i,:])),np.argmin(np.diff(image[i,:]))) for i in row_rng ]
        
        m = [i[0]+((i[1]-i[0])/2) for i in lr]
        a,b = np.polyfit(row_rng,m,1)
        mf = [a*i+b for i in row_rng]
        cr = int(rows/2)
        cc = mf[cr]
        degree = np.arctan(a)/np.pi*180
        
        dst = scipy.ndimage.rotate(image,-degree)
        cols_dst = dst.shape[1]
        df = cols_dst - cols
        dst = dst[:,df:cols_dst - df]
        dst = dst - np.min(dst)
        dst = dst / 10000 # micrometer
        rng = np.arange(dst.shape[1]) * 2798.716 / 1000
        mean = np.mean(dst,axis=0)
        li,ri,d,s = triangle(mean,left=0,right=len(mean)-1)
        amx = np.argmax(mean)
        p,e = scipy.optimize.curve_fit(self.piecewise_function,rng,mean,[li,li+(amx-li)/2,ri-(ri-amx)/2,ri,0.1,1.0,0.1,-1.0])
        mean_fit = self.piecewise_function(rng,*p)
        std = np.std(dst,axis=0)

        l = [i[0] for i in lr]
        r = [i[1] for i in lr]
        
        view_data = {"I":dst,"l":[m,row_rng],"r":[mf,row_rng],
                     "mean":[rng,mean],"std":[rng,mean_fit],
                     "left":0,"right":0}
        pass_data = []
        
        frame.addData(self.getName(),pass_data,view_data)
        
    def setParams(self,par_list):
        name = par_list[0]
        value = par_list[1] 
        
        if name == "row":
            self.row = value
        elif name == "degree":
            self.degree = value / 10

class Calibration_Keyence(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Calibration Keyence"                                  
    __VIEW__ = [["Height Map","column","row",0,0,6,8,    
                             {"I":"image",
                              "l":"curve",
                              "r":"curve"}],
                ["Row","row","height",6,0,3,8,
                             {"mean":"curve",
                              "std":"curve",
                              "left":"vline",
                              "right":"vline"}],              
                             ]

    __SETS__ = [["Intersection",[["Row","L",0,0,1,1],
                                 ["row2","S",0,1,1,1,0,0,5000],
                                 ["degree","S",1,0,1,1,0,0,3600],
                                 ["transpose","S",2,0,1,1,0,0,1]
                                 ]
                        ]
                ]
                
    def initialize(self):    
        self.row = 0
        self.degree = 1
        self.transpose = 0
        
    def getWidth(self,i,img,left,right):
        col = img[i,:].flatten()
        li,ri,d,s = triangle(col,left=left,right=right)
        
        if verify(col,li,ri,left,right):
            return (li,ri)
        else:
            return (0,0)    
       
    def getHeight(self,i,img):
        col = img[:,i].flatten()  
        height_nanometer = float(max(col) ) /10
        height_mm = height_nanometer / 1000000
        return ( height_mm,np.argmax(col) )
        
    def piecewise_function(self,x,lb,lt,rt,rb,b0,a1,a2,a3):
        cond_list = [x < lb,(x>=lb) & (x<lt),(x>=lt) & (x<=rt),(x>rt) & (x<=rb),x>rb]

        func_list  = [lambda x: b0, 
                   lambda x: b0+ a1*(x-lb), 
                   lambda x: b0+ a1*(x-lb) + a2*(x-rb), 
                   lambda x: b0+ a1*(x-lb) + a2*(x-lt) + a3*(x-rt),
                   lambda x: b0]
        
        return np.piecewise(x,cond_list,func_list)
    
    def transform(self,frame):
        image1 = frame.getLastPassData()[0]
        
        if self.transpose:
            image = image1.transpose().copy()
        else:
            image = image1
            
        s = image.shape
        
        rows = s[0]
        cols = s[1]
        
        row_rng = np.arange(rows)
        col_rng = np.arange(cols)
        
        #lr = [self.getWidth(i,image,left=0,right=cols-1) for i in row_rng]
        lr = [(np.argmax(np.diff(image[i,:])),np.argmin(np.diff(image[i,:]))) for i in row_rng ]
        
        m = [i[0]+((i[1]-i[0])/2) for i in lr]
        a,b = np.polyfit(row_rng,m,1)
        mf = [a*i+b for i in row_rng]
        cr = int(rows/2)
        cc = mf[cr]
        degree = np.arctan(a)/np.pi*180
        
        dst = scipy.ndimage.rotate(image,-degree)
        cols_dst = dst.shape[1]
        df = cols_dst - cols
        dst = dst[:,df:cols_dst - df]
        dst = dst - np.min(dst)
        dst = dst / 10000 # micrometer
        rng = np.arange(dst.shape[1]) * 2798.716 / 1000
        mean = np.mean(dst,axis=0)
        li,ri,d,s = triangle(mean,left=0,right=len(mean)-1)
        amx = np.argmax(mean)
        p,e = scipy.optimize.curve_fit(self.piecewise_function,rng,mean,[li,li+(amx-li)/2,ri-(ri-amx)/2,ri,0.1,1.0,0.1,-1.0])
        mean_fit = self.piecewise_function(rng,*p)
        std = np.std(dst,axis=0)

        l = [i[0] for i in lr]
        r = [i[1] for i in lr]
        
        view_data = {"I":image,"l":[m,row_rng],"r":[mf,row_rng],
                     "mean":[rng,mean],"std":[rng,mean_fit],
                     "left":0,"right":0}
        pass_data = []
        
        frame.addData(self.getName(),pass_data,view_data)
        
    def setParams(self,par_list):
        name = par_list[0]
        value = par_list[1] 
        print(name,value)
        if name == "row":
            self.row = value
        elif name == "degree":
            self.degree = value / 10
        elif name == "transpose":
            self.transpose = value
            
class Intersection_Keyence(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Keyence - Intersection"                                  
    __VIEW__ = [["Height Map","column","row",0,0,6,8,    
                             {"I":"image"}],
                ["Intersection","column","height",6,0,3,8,
                             {"sec":"curve"}],              
                             ]

    __SETS__ = [["Intersection",[["Transpose","L",0,0,1,1],
                                 ["transpose","S",0,1,1,1,0,0,1],
                                 ]
                        ]
                ]
                
    def initialize(self):    
        self.angle = 0
        self.transpose = 0

    def rotateAndScale(self,img, scaleFactor = 0.5, degreesCCW = 30):
        (oldY,oldX) = img.shape #note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
        M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=degreesCCW, scale=scaleFactor) #rotate about center of image.
    
        #choose a new image size.
        newX,newY = oldX*scaleFactor,oldY*scaleFactor
        #include this if you want to prevent corners being cut off
        r = np.deg2rad(degreesCCW)
        newX,newY = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),abs(np.sin(r)*newX) + abs(np.cos(r)*newY))
    
        #the warpAffine function call, below, basically works like this:
        # 1. apply the M transformation on each pixel of the original image
        # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.
    
        #So I will find the translation that moves the result to the center of that region.
        (tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
        M[0,2] += tx #third column of matrix holds translation, which takes effect after rotation.
        M[1,2] += ty
    
        rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX),int(newY)))
        return rotatedImg
    
    def totalSTD(self,angle,image):
        image2 = self.rotateAndScale(image,scaleFactor=1.0,degreesCCW=angle)

        image2[image2==0] = np.nan
        std = np.nanstd(image2,axis=0)
        return np.nansum(std)
        
    def transform(self,frame):
        csv_dict = frame.getLastPassData()[0]
        
        image1 = csv_dict["Data"] * float(csv_dict["Z calibration"][0]) / 1000
        
        if self.transpose:
            image = image1.transpose().copy()
        else:
            image = image1
            
            
        res = minimize_scalar(self.totalSTD,args=(image,),method="golden")
        rotated = self.rotateAndScale(image,scaleFactor=1.0,degreesCCW=res.x)

        mean = np.nanmean(rotated,axis=0)
        rng = np.arange(len(mean)) * float(csv_dict["XY calibration"][0]) / 1000
        
        view_data = {"I":rotated,"sec":[rng,mean]}
        pass_data = []
        
        frame.addData(self.getName(),pass_data,view_data)
        
    def setParams(self,par_list):
        name = par_list[0]
        value = par_list[1] 

        if name == "row":
            self.row = value
        elif name == "angle":
            self.angle = value / 100
        elif name == "transpose":
            self.transpose = value
        elif name == "canny_low":
            self.canny_low = value
        elif name == "canny_high":
            self.canny_high = value
        elif name == "hough_low":
            self.hough_low = value
        elif name == "nr_lines":
            self.nr_lines = value
        elif name == "left":
            self.left = value
        elif name == "right":
            self.right = value
            
class Homography_Estimate(Transformation):
    
    __INST__ = 0   
    __NAME__ = "Homography Estimation"                                  
    __VIEW__ = [["Height Map","column","row",0,0,6,8,    
                             {"I":"image",
                              "row":"curve"}],
                ["Intersection","column","height",6,0,3,8,
                             {"sec":"curve"}],              
                             ]

    __SETS__ = [["Intersection",[["Transpose","L",0,0,1,1],
                                 ["transpose","S",0,1,1,1,0,0,1],
                                 ["Row","L",1,0,1,1],
                                 ["row","S",1,1,1,1,0,0,5000],
                                 ["Angle","L",2,0,1,1],
                                 ["angle","S",2,1,1,1,0,0,90],
                                 ]
                        ]
                ]
                
    def initialize(self):    
        self.row = 0
        self.angle = 0
        self.transpose = 0
    
    def transform(self,frame):
        image1 = frame.getLastPassData()[0]
        
        if self.transpose:
            image = image1.transpose().copy()
        else:
            image = image1
            
        s = image.shape
        
        rows = s[0]
        cols = s[1]
        
        row_rng = np.arange(rows)
        col_rng = np.arange(cols)
        
        #lr = [self.getWidth(i,image,left=0,right=cols-1) for i in row_rng]
        lr = [(np.argmax(np.diff(image[i,:])),np.argmin(np.diff(image[i,:]))) for i in row_rng ]
        
        m = [i[0]+((i[1]-i[0])/2) for i in lr]
        a,b = np.polyfit(row_rng,m,1)
        mf = [a*i+b for i in row_rng]
        cr = int(rows/2)
        cc = mf[cr]
        degree = np.arctan(a)/np.pi*180
        
        dst = scipy.ndimage.rotate(image,-degree)
        cols_dst = dst.shape[1]
        df = cols_dst - cols
        dst = dst[:,df:cols_dst - df]
        dst = dst - np.min(dst)
        dst = dst / 10000 # micrometer
        rng = np.arange(dst.shape[1]) * 2798.716 / 1000
        mean = np.mean(dst,axis=0)
        li,ri,d,s = triangle(mean,left=0,right=len(mean)-1)
        amx = np.argmax(mean)
        p,e = scipy.optimize.curve_fit(self.piecewise_function,rng,mean,[li,li+(amx-li)/2,ri-(ri-amx)/2,ri,0.1,1.0,0.1,-1.0])
        mean_fit = self.piecewise_function(rng,*p)
        std = np.std(dst,axis=0)

        l = [i[0] for i in lr]
        r = [i[1] for i in lr]
        
        view_data = {"I":image,"row":self.row,"r":[mf,row_rng],
                     "mean":[rng,mean],"std":[rng,mean_fit],
                     "left":0,"right":0}
        pass_data = []
        
        frame.addData(self.getName(),pass_data,view_data)
        
    def setParams(self,par_list):
        name = par_list[0]
        value = par_list[1] 
        print(name,value)
        if name == "row":
            self.row = value
        elif name == "degree":
            self.degree = value / 10
        elif name == "transpose":
            self.transpose = value