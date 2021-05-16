
## Functions to evaluate performance ##

import re

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())
