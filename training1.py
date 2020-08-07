"""
Train the schema matching model and save it as matcher1.model.

This module pre-process the training data, summerize the mappings in the training data (knowledge base) and use the matcher class to generate model.

"""
import os
import json
import pandas as pd
import pickle
import ln_matcher

schema_list = []
mapping_list = []

path_training = "./data/training/"
with os.scandir(path_training) as it:
    for entry in it:
        if entry.name.endswith(".csv") and entry.is_file():
            #print(entry.name, entry.path)
            data = pd.read_csv(entry.path)
            # pre-processing input data, remove columns with more than half missing values
            data = data.dropna(thresh=len(data.index)/2, axis='columns')
            # pre-processing input data, remove columns with empty name
            data = data[data.columns.dropna()]  # drop col w/o no headerm           
            data.columns = data.columns.str.strip()     # remove space b/a header 
            data = data.loc[:, data.tail(1).notnull().all()]    # drop col w/o label
            # capitalize all the labels 
            for i in range(len(data.iloc[-1,:])):
                s = data.iloc[-1,i]
                lst = [word[0].upper() + word[1:] for word in s.split()]
                data.iloc[-1,i] = " ".join(lst)

            train = data.drop(data.tail(1).index)
            train = train.astype(str)   # convert all columns to str, mod 7/8rip()
            
            map = data.tail(1).to_dict('list')
            map = {i:str(j[0]) for i,j in map.items() }
            schema_list.append(train)
            mapping_list.append(map)

# Collect the existing mappings in the trianing data
mapping_dict = dict(j for i in mapping_list for j in i.items())
knowledge_base = {}
for key, value in mapping_dict.items():
    knowledge_base.setdefault(value, set()).add(key)
for key in knowledge_base:
    print(key, ' = ', knowledge_base[key])

# Using ln_matcher
fm = ln_matcher.ln_matcher(schema_list, mapping_list, sample_size=100)
fm.train()                                           # train model
fm.save_model('matcher1')


