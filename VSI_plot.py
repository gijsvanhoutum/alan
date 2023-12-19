# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 11:30:21 2018

@author: Gijs van Houtum
"""
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



class Plotter(QDialog):
    
    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)

        self.buttonBox = QtGui.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)

        self.textBrowser = QtGui.QTextBrowser(self)
        self.textBrowser.append("This is a QTextBrowser!")

        self.verticalLayout = QtGui.QVBoxLayout(self)
        self.verticalLayout.addWidget(self.textBrowser)
        self.verticalLayout.addWidget(self.buttonBox)
        

class Plotter: 
    
    def __init__(self,ll_specs):

        self.ll_specs = ll_specs     
        self.d_data = None
        self.i_nr_rows = None
        self.i_nr_cols = None
        
        self.gs = self._setGridSize()

    def _setGridSize(self):
        
        i_nr_rows = 0
        i_nr_cols = 0
        
        for l_spec in self.ll.specs:
            i_start_row = l_spec[2]
            i_start_col = l_spec[3]
            i_row_span = l_spec[4]
            i_col_span = l_spec[5]
            
            i_max_row = i_start_row + i_row_span
            i_max_col = i_start_col + i_col_span
            
            if i_max_row > i_nr_rows:
                i_nr_rows = i_max_row
                
            if i_max_col > i_nr_cols:
                i_nr_cols = i_max_col
                
        return gridspec.GridSpec(i_nr_rows, i_nr_cols)
        
    def setData(self, d_data):
        self.d_data = d_data
            
    def plot(self):
        self.gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 
 
        fig = plt.gcf()
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        
        for l_spec in self.ll_specs:
            i_start_row = l_spec[2]
            i_start_col = l_spec[3]
            i_row_span = l_spec[4]
            i_col_span = l_spec[5]
            s_xlabel = l_spec[6]
            s_ylabel = l_spec[7]
            s_visible = l_spec[8]
            
            spec = self.gs[i_start_row : i_start_row + i_row_span,
                           i_start_col : i_start_col + i_col_span]
                      
            print type(spec)
            
            ax = plt.subplot(spec)
            ax.autoscale(axis='x',tight=True)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            if visible == "Y":
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(True)
                
            elif visible == "X":
                ax.get_xaxis().set_visible(True)
                ax.get_yaxis().set_visible(False)
            elif visible == "YX":
                ax.get_xaxis().set_visible(True)
                ax.get_yaxis().set_visible(True)
                
            name = s[0]
            view = self.views[name]
            data = view.getData()     
            
            if s[1] == "2D":   
                for i in range(len(data)):
                    ax.plot(data[i]['x'],data[i]['y'])
               
            elif s[1] == "3D":
                ax.imshow(data)

        fig.set_size_inches(width,height)
        fig.savefig('test2png.pdf') 

        plt.show()        
        
    def save(self):
        nr_rows = self.grid.rowCount()
        nr_cols = self.grid.columnCount()

        gs = gridspec.GridSpec(nr_rows, nr_cols)
        
        
        image_width = 2592
        image_height = 1944
        image_ratio = float(image_height) / image_width

        pad = 0.1
        width = 8.27 # inches
        height = image_ratio*width
        
        
        fig = plt.gcf()
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        


    def fixed_aspect_ratio(self, axis, ratio):
        '''
        Set a fixed aspect ratio on matplotlib plots 
        regardless of axis units
        '''
        xvals,yvals = axis.get_xlim(),axis.get_ylim()
        axis.set_adjustable('box')
        xr = xvals[1]-xvals[0]
        yr = yvals[1]-yvals[0]
        axis.set_aspect(ratio*(xr/float(yr)))