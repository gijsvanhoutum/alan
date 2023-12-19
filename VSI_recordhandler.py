


class RecordHandler:
    def __init__(self,record_class):
        
        self.record_class = record_class
        self.status = None
        
        self.frames = 0
        self.times = []
        
        self.codec = 'XVID'
        self.video_ext = '.avi'
        self.data_ext = ".josh"
        self.record_dir = "./recordings/"
        self.temp_filename = "temp"
        
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)
       
        self.fps = 1000
        self.writer = None
        
    def addFrame(self,frame):
        if self.status is "run":
            image = frame.getSaveImage()
            if self.writer is None:
                self.setupWriter(image)

            image = cv2.resize(image,(self.width,self.height))
            self.writer.write(image)
            self.times.append(frame.getTime())
            self.frames+=1
    
    def setupWriter(self,image):
        self.width = image.shape[1]
        self.height = image.shape[0]
        frame_size = (self.width,self.height)
        fourcc = cv2.cv.CV_FOURCC(*self.codec)

        self.writer = cv2.VideoWriter(self.record_dir+self.temp_filename+\
                                        self.video_ext,fourcc,self.fps,
                                        frame_size,isColor=True)
                      
    def startRecording(self):
        self.status = "run"
              
    def stopRecording(self):
        if self.status is "run":
            self.status = "stop"
            self.writer.release()
            self.writer = None

    def saveRecording(self,filename,video_info):
        for name in os.listdir(self.record_dir):
            if name == self.temp_filename+self.video_ext:
                old_video_name = self.record_dir+name
                new_video_name = filename+self.video_ext
                os.rename(old_video_name,new_video_name)
        
        record = self.record_class(video_info,self.times,new_video_name)
                
        with open(filename+self.data_ext,"wb") as f:
            pickle.dump(record,f,pickle.HIGHEST_PROTOCOL)
    
    def getDirectory(self):
        return self.record_dir
        
    def getExtension(self):
        return self.data_ext
        
    def getNrFrames(self):
        return len(self.frames)