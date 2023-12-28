import time

class Frame:
    def __init__(self):
        
        self.position = None
        
        self.time = {}
        self.pass_data = {}
        self.view_data = {}
        
        self.nameorder = []
        
        self.lastname = None
        
    def setPosition(self,position):
        self.position = position
        
    def getPosition(self):
        return self.position
    
    def getTimeDiffSeconds(self,name1, name2):
        time1 = self.time[name1]
        time2 = self.time[name2]
        
        return max(time1-time2,time2-time1)

    def getLastPassData(self):
        return self.pass_data[self.lastname]
        
    def getLastViewData(self):
        try:
            return self.view_data[self.lastname]
        except:
            return None
        
    def getPassData(self,name):
        return self.pass_data[name]
        
    def getViewData(self,name):
        try:
            return self.view_data[name]
        except:
            return None
    
    def addData(self,name,pass_data,view_data):

        self.view_data[name] = view_data
        self.pass_data[name] = pass_data
        self.time[name] = time.time()   
        self.nameorder.append(name)
        self.lastname = name
        
        
