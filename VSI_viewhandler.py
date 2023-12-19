from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class ViewHandler(QSplitter):
    
    command = pyqtSignal(int)
    
    def __init__(self, view_class, parent=None):
        super(self.__class__, self).__init__(parent)
    
        self.view_class = view_class
        self.paused = False      
    
    def _createTabWidget(self):
        tabwidget = QTabWidget()      
        tabwidget.setTabsClosable(True)  
        tabwidget.currentChanged.connect(self._changed)
        tabwidget.setAutoFillBackground(True)

        saveButton = QPushButton()  
        saveButton.setIcon(QIcon.fromTheme("document-save-as"))
        tabwidget.setCornerWidget(saveButton)
        
        l = lambda: self._saveView(tabwidget.currentWidget())
        saveButton.clicked.connect(l)
        tabwidget.tabCloseRequested.connect(self._removeView)    
        self.addWidget(tabwidget) 
        return tabwidget
    
    def _changed(self,index):
        
        policy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        for ti in range(self.count()):
            tabwidget = self.widget(ti)
            tabwidget.widget(index).setSizePolicy(policy)  
  
    def _removeTabWidget(self,index):
        widget = self.widget(index)
        widget.hide()
        widget.deleteLater()
                       
    def _removeView(self,index):  
        nr_vw = self.widget(0).count()
        if nr_vw > 1:
            for idx in range(self.count()):
                old = self.widget(idx)
                wg = old.widget(index)
                wg.hide()
                wg.deleteLater()
        else:
            self.quitViews()

        self.command.emit(index)

    def _saveView(self, view):
        view.save()
        
    def addView(self,name,specs):
        if specs is not None:
            cnt = self.count()
            if cnt == 0:
                self._createTabWidget()        
                cnt+=1
                
            for idx in range(cnt):
                view = self.view_class(name,specs)    
                tabw = self.widget(idx)
                tabw.addTab(view,name) 
       
    def startViews(self):         
        self.paused = False
                
    def pauseViews(self):
        self.paused = True

    def quitViews(self):
        for i in range(self.count()):
            self._removeTabWidget(i)

    def splitViews(self):
        tabwidget = self._createTabWidget()  
        
        tw = self.widget(0)
        for idx in range(tw.count()): 
            widget = tw.widget(idx)
            name = widget.getName()
            specs = widget.getSpecs()
            copy_view = self.view_class(name,specs)
            tabwidget.addTab(copy_view,name)
     
        self.addWidget(tabwidget)
            
    def mergeViews(self):
        c = self.count()
        if c > 1:
            self._removeTabWidget(c-1)        
   
    def updateViews(self,frame):
        if not self.paused:
            for tw_idx in range(self.count()):
                tabwidget = self.widget(tw_idx)
                for vw_idx in range(tabwidget.count()):
                    view = tabwidget.widget(vw_idx)
                    
                    try:
                        name = view.getName()
                        data = frame.getViewData(name)
                        if data is not None:
                            view.setData(data)
                    except:
                        pass