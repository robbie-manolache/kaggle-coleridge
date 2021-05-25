
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
    
    def random_text_ids(self, n=10):
        n = min(len(self.text_ids), n)        
        self.sample_ids = list(np.random.choice(self.text_ids, n))
        
    def load_text_from_id_list(self, id_list="random", n=10):
        
        # set id_list 
        if id_list == "random":
            if self.sample_ids is None:
                self.random_text_ids(n)         
        elif id_list == "full":
            self.sample_ids = self.text_ids       
        else:
            self.sample_ids = id_list
            
        # load each text file
        text_dict = {}
        for text_id in self.sample_ids:
            with open(os.path.join(self.text_dir, text_id+".json")) as rf:
                text_dict[text_id] = json.load(rf)
        self.text_dict = text_dict
        
    def select_sample_text(self):
        
        if self.text_dict is None:
            self.load_text_from_id_list(n=1)
            
        id_list = list(self.text_dict.keys())
        sample_id = list(np.random.choice(id_list, 1))[0]
        
        return sample_id, self.text_dict[sample_id]
    
    def reset_id_selection(self):
        self.sample_ids = None
        self.text_dict = None
       