# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:31:08 2017

@author: darren
"""

import numpy as np
import glob
from PIL import Image
import random

random.seed(60)

class Process_data(object):
    
    def __init__(self):
        self.set1_path = "set1_lib81_1_smoke552_non831"
        self.set2_path = "set2_lib81_2_smoke688_non817"
        self.set3_path = "set3_set_smoke2201_non8511"
        self.set4_path = "set4_set_smoke2254_non8363"    
        self.small_path = "small_set_50"
        self.smoke_label = np.array([1.0,0.0])
        self.nonsmoke_label = np.array([0.0,1.0])
        
    def load_data(self,path,label):
        
        #put all of the data name in the data list
        data_list=glob.glob(path)
        data=[]    
        
        for i in range(len(data_list)):
            
            im = Image.open(data_list[i])  
            resize_im = im.resize((64,64))            
            im_array = np.array(resize_im,dtype=np.float64)/255
            #im_array -= np.mean(im_array, axis = 0) # zero-center            
            #im_array /= np.std(im_array, axis = 0) # normalize
            data.append((im_array,label))
            
        return data
    
    def creat_train_test_data(self):        
        
        
        set1_smoke_data = self.load_data(self.set1_path+"/smoke/*.jpg",self.smoke_label)
        set1_nonsmoke_data = self.load_data(self.set1_path+"/non/*.jpg",self.nonsmoke_label)    
        
        set2_smoke_data = self.load_data(self.set2_path+"/smoke/*.jpg",self.smoke_label)
        set2_nonsmoke_data = self.load_data(self.set2_path+"/non/*.jpg",self.nonsmoke_label)
        
        set3_smoke_data = self.load_data(self.set3_path+"/smoke/*.jpg",self.smoke_label)
        set3_nonsmoke_data = self.load_data(self.set3_path+"/non/*.jpg",self.nonsmoke_label)
        
        set4_smoke_data = self.load_data(self.set4_path+"/smoke/*.jpg",self.smoke_label)
        set4_nonsmoke_data = self.load_data(self.set4_path+"/non/*.jpg",self.nonsmoke_label)
        
        small_smoke_data = self.load_data(self.small_path+"/smoke/*.jpg",self.smoke_label)
        small_nonsmoke_data = self.load_data(self.small_path+"/smoke/*.jpg",self.nonsmoke_label)
        train_data=[]
        train_data.extend(set3_smoke_data)
        train_data.extend(set3_nonsmoke_data)
        train_data.extend(set4_smoke_data)
        train_data.extend(set4_nonsmoke_data)
        train_data.extend(set1_smoke_data)
        train_data.extend(set1_nonsmoke_data)
        train_data.extend(set2_smoke_data)
#        train_data.extend(small_nonsmoke_data)
        
        test_data=[]
        test_data.extend(small_smoke_data)
        test_data.extend(small_nonsmoke_data)
        
        random.shuffle(train_data)
        random.shuffle(test_data)
        
        print ("Data process complete:")
        print ("training data:",len(train_data))
        print ("testing data:",len(test_data))
        return train_data,test_data


#train_data,test_data=Process_data().creat_train_test_data()
#pass