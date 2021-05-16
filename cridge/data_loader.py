
## Functions to load and pre-process text files ##

import os
import json
import numpy as np
import pandas as pd

class Coleridger:
    
    def __init__(self, comp_dir, data_type="train"):
        
        # initalise text directory and ids
        text_dir = os.path.join(comp_dir, data_type)
        self.text_dir = text_dir
        self.text_ids = [t[:-5] for t in os.listdir(text_dir)]
        self.text_dict = None
        self.sample_ids = None
        
        # load df with labels if available
        if data_type == "train":
            self.text_df = pd.read_csv(os.path.join(comp_dir, "train.csv"))
        else:
            self.text_df = None
    
    def sample_text_ids(self, n=10):
        n = min(len(self.text_ids), n)        
        self.sample_ids = list(np.random.choice(self.text_ids, n))
        
    def load_text_from_id_list(self, id_list="sample", n=10):
        
        # set id_list 
        if id_list == "sample":
            if self.sample_ids is None:
                self.sample_text_ids(n)
            id_list = self.sample_ids           
        elif id_list == "full":
            id_list = self.text_ids       
        else:
            id_list = id_list
            
        # load each text file
        text_dict = {}
        for text_id in id_list:
            with open(os.path.join(self.text_dir, text_id+".json")) as rf:
                text_dict[text_id] = json.load(rf)
        self.text_dict = text_dict
        
    def select_sample_text(self):
        
        if self.text_dict is None:
            self.load_text_from_id_list(n=1)
            
        id_list = list(self.text_dict.keys())
        sample_id = list(np.random.choice(id_list, 1))[0]
        
        return sample_id, self.text_dict[sample_id]
       