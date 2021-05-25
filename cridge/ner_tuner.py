
## Functions for tuning spacy NER model ##

import os
import re
import json
import spacy
from tqdm import tqdm
from cridge.data_loader import Coleridger
from cridge.data_maker import refine_texts

def __span_overlap__(span1, span2):
    """
    """
    if (span1[0] <= span2[1]) & (span2[0] <= span1[1]):
        return True
    else:
        return False

def __unique_spans__(labels, text):
    """
    """
    
    # iterate through all labels
    label_locs = []
    for l in labels:
        
        # add expected punctuation regex and look for matches
        rgx = "[\s|\n|(]" + l + "[\s|,|.|?|!|:|;|)]"
        for m in re.finditer(rgx, " "+text+" "):
            
            # get span of matches and adjust for text space offset
            ll_new = m.span()
            ll_new = (ll_new[0], ll_new[1]-2)
            add_new = True
            
            # check if there's an overlapping span and decide which to add
            for i, ll in enumerate(label_locs):
                if __span_overlap__(ll_new, ll):
                    add_new = False
                    if (ll_new[1] - ll_new[0]) > (ll[1] - ll[0]):
                        label_locs[i] = ll_new
            if add_new:
                label_locs.append(ll_new)

    return label_locs

class NER_tuner(Coleridger):
    """
    """
    
    def __init__(self, comp_dir, 
                 data_type="train", 
                 model="en_core_web_sm", 
                 proc_text_file=None,
                 tune_text_file=None):
        """
        """
        
        super().__init__(comp_dir, data_type=data_type)
        self.nlp = spacy.load(model)
        
        # set path to processed texts
        if proc_text_file is None:
            self.proc_text_path = None
        else:
            self.proc_text_path = os.path.join(comp_dir, proc_text_file)
            
        # set path to NER tuning texts
        if tune_text_file is None:
            self.tune_text_path = None
        else:
            self.tune_text_path = os.path.join(comp_dir, tune_text_file)
    
    def refine_train_texts(self, id_list="random", n=10, auto_reset=True,
                           sent_spans=[(2,2), (3,3)]):
        """
        """
        
        # reset sample ids
        if auto_reset:
            self.reset_id_selection()
        
        # get list of ids and labels
        self.load_text_from_id_list(id_list=id_list, n=n)
        label_df = self.text_df.query("Id in @self.sample_ids")
        
        # load any processed texts
        if self.proc_text_path is None:
            print("Please specify a path for processed text!")
            return None
        else:
            if os.path.exists(self.proc_text_path):
                with open(self.proc_text_path) as rf:
                    train_data = json.load(rf)
                procd_ids = list(train_data.keys())
            else:
                train_data = {}
                procd_ids = None
        
        # refine training texts
        train_data_new = refine_texts(label_df, self.text_dict,
                                      sent_spans, self.nlp, 
                                      procd_ids=procd_ids)
        
        # update set of processed texts
        train_data.update(train_data_new)
        
        # save to disk
        with open(self.proc_text_path, "w") as wf:
            json.dump(train_data, wf)
    
    def load_refined_texts(self):
        if os.path.exists(self.proc_text_path):
            with open(self.proc_text_path) as rf:
                proc_texts = json.load(rf)
        return proc_texts
        
    def gen_train_data(self):
        """
        """
        
        # load any processed texts
        if self.proc_text_path is None:
            print("Please specify a path for processed text!")
            return None
        else:
            if os.path.exists(self.proc_text_path):
                with open(self.proc_text_path) as rf:
                    proc_texts = json.load(rf)
            else:
                print("Please refine texts before creating NER tuning text!")
                return None
            
        # load any NER tuning texts
        if self.tune_text_path is None:
            print("Please specify a path for NER tuning text!")
            return None
        else:
            if os.path.exists(self.tune_text_path):
                with open(self.tune_text_path) as rf:
                    tune_data = json.load(rf)
                    tune_ids = list(tune_data.keys())
            else:
                tune_data = {}  
                tune_ids = None      
        
        # get list of applicable ids and their labels
        id_list = list(proc_texts.keys())    
        if tune_ids is not None:
            id_list = [i for i in id_list if i not in tune_ids]                 
        label_df = self.text_df.query("Id in @id_list")
        
        # iterate through each id to compile training data
        tune_data_new = {}
        for pub_id in tqdm(id_list):
            
            # get text and labels
            pub_text = proc_texts[pub_id]
            labels = label_df.query("Id == @pub_id")["dataset_label"].tolist()
            pub_train = []
            
            # iterate through each section
            for sec in pub_text:
                if any([s in sec for s in labels]):
                    
                    # process text and add dataset entities first
                    doc = self.nlp(sec)
                    ent_dict = {"entities": []}
                    label_locs = __unique_spans__(labels, sec)
                    for ll in label_locs:
                        ent_dict["entities"].append((ll[0], ll[1], 
                                                     "DATASET"))
                        
                    # add all remaining entities (excluding datasets)
                    for e in doc.ents:
                        add_example = True
                        for ll in label_locs:
                            if __span_overlap__((e.start_char, 
                                                 e.end_char), ll):
                                add_example = False
                        if add_example:
                            ent_dict["entities"].append((e.start_char, 
                                                         e.end_char, 
                                                         e.label_))
                    pub_train.append((doc.text, ent_dict))
                    
            tune_data_new[pub_id] = pub_train
        
        # update set of processed texts
        tune_data.update(tune_data_new)
        
        # save to disk
        with open(self.tune_text_path, "w") as wf:
            json.dump(tune_data, wf) 
            
    def load_tuning_texts(self):
        if os.path.exists(self.tune_text_path):
            with open(self.tune_text_path) as rf:
                tune_texts = json.load(rf)
        return tune_texts
