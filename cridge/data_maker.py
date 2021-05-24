
## Functions for extracting text excerpts for training ##

import pandas as pd

def __compile_from_sent_list__(start, end, sent_list):
    return " ".join([s.text for s in sent_list[start:end]])

def refine_texts(label_df, raw_texts, sent_spans, nlp,
                 procd_ids=None):
    """
    """
    
    # extract list of ids and initalize empty dict to collect data
    id_list = label_df["Id"].unique().tolist()
    if procd_ids is not None:
        id_list = [i for i in id_list if i not in procd_ids]
    train_dict = {}
    
    # iterate through all ids
    for pub_id in id_list:
        
        # get raw text and dataset labels
        pub_text = raw_texts[pub_id]
        labels = label_df.query("Id == @pub_id")["dataset_label"].tolist()
        
        # compile training data from each text section
        train_data = []
        for sec in pub_text:
            
            # if the text contains the label
            if any([s in sec["text"] for s in labels]):
                
                # process with spacy and check if sentenced
                doc = nlp(sec["text"])
                if doc.is_sentenced:
                    
                    # collect sentence groups
                    sent_tups = []
                    all_sents = list(doc.sents)
                    
                    # iterate through all sentences
                    for i, sent in enumerate(doc.sents):
                        
                        # pick sentence with the label
                        if any([s in sent.text for s in labels]):
                            
                            # get sentence indices surrounding label
                            for span in sent_spans:
                                sent_tups.append((i, 
                                                  max((i - span[0]), 0), 
                                                  min((i + span[1] + 1), 
                                                      len(all_sents))))
                                
                    # compile sentence index dataframe
                    sent_df = pd.DataFrame(sent_tups, 
                                           columns=["sent_idx", 
                                                    "min_idx", 
                                                    "max_idx"])
                    
                    # compile texts from selected sentences
                    train_texts = sent_df.apply(
                        lambda x: __compile_from_sent_list__(
                            start=x["min_idx"], 
                            end=x["max_idx"],
                            sent_list=all_sents), axis=1).tolist()
                    
                    # add to train data for current id
                    train_data += train_texts
                    
        train_dict[pub_id] = train_data
        
    return train_dict
