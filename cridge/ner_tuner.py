
## Functions for tuning spacy NER model ##

import re
import spacy
from cridge.data_loader import Coleridger

class NER_tuner(Coleridger):
    """
    """
    
    def __init__(self, comp_dir, data_type="train", 
                 model="en_core_web_sm"):
        """
        """
        
        super().__init__(comp_dir, data_type=data_type)
        self.nlp = spacy.load(model)
        
    def gen_train_data(self, id_list="random", n=10):
        
        self.load_text_from_id_list(id_list=id_list, n=n)
        label_df = self.text_df.query("Id in @self.sample_ids")
        
        train_data = []
        for pub_id in self.sample_ids:
            pub_text = self.text_dict[pub_id]
            labels = label_df.query("Id == @pub_id")["dataset_label"].tolist()
            for sec in pub_text:
                if any([s in sec["text"] for s in labels]):
                    doc = self.nlp(sec["text"])
                    ent_dict = {"entities": []}
                    label_locs = [m.span() for l in labels for 
                                  m in re.finditer(l, sec["text"])]
                    for ll in label_locs:
                        ent_dict["entities"].append((ll[0], ll[1], 
                                                     "DATASET"))
                    for e in doc.ents:
                        add_example = True
                        for ll in label_locs:
                            if e.start_char < ll[1] & e.end_char > ll[0]:
                                add_example = False
                        if add_example:
                            ent_dict["entities"].append((e.start_char, 
                                                         e.end_char, 
                                                         e.label_))
                    train_data.append((doc.text, ent_dict))
                    
        self.train_data = train_data 
