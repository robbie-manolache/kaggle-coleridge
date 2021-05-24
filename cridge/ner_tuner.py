
## Functions for tuning spacy NER model ##

import os
import re
import json
import spacy
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
    label_locs = []
    for l in labels:
        for m in re.finditer(l, text):
            ll_new = m.span()
            add_new = True
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
    
    def __init__(self, comp_dir, data_type="train", 
                 model="en_core_web_sm", proc_text_file=None):
        """
        """
        
        super().__init__(comp_dir, data_type=data_type)
        self.nlp = spacy.load(model)
        if proc_text_file is None:
            self.proc_text_path = None
        else:
            self.proc_text_path = os.path.join(comp_dir, proc_text_file)
    
    def refine_train_texts(self, id_list="random", n=10,
                           sent_spans=[(2,2), (3,3)]):
        
        # get list of ids and labels
        self.load_text_from_id_list(id_list=id_list, n=n)
        label_df = self.text_df.query("Id in @self.sample_ids")
        
        # load any processed texts
        if self.proc_text_path is None:
            print("Please specify a path for processed data!")
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
        
        # save and return
        with open(self.proc_text_path, "w") as wf:
            json.dump(train_data, wf)
        return train_data
        
    def gen_train_data(self):
        
        # load any processed texts
        if self.proc_text_path is None:
            print("Please specify a path for processed data!")
            return None
        else:
            if os.path.exists(self.proc_text_path):
                with open(self.proc_text_path) as rf:
                    train_texts = json.load(rf)
            else:
                print("Please refine texts before creating training data!")
                return None
        
        # get list of applicable ids and their labels
        procd_ids = list(train_texts.keys())                     
        label_df = self.text_df.query("Id in @procd_ids")
        
        # iterate through each id to compile training data
        train_data = []
        for pub_id in procd_ids:
            
            # get text and labels
            pub_text = train_texts[pub_id]
            labels = label_df.query("Id == @pub_id")["dataset_label"].tolist()
            
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
                    train_data.append((doc.text, ent_dict))
        
        # build training data            
        self.train_data = train_data 
