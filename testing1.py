"""
Implement matcher1.model.

This module feeds the model with input data at path_testing. Output json files are saved at path_ouput.

"""
import os
import json
import pandas as pd
import pickle
import ln_matcher

path_testing = "./data/testing/"
path_output = "./data/output/"
with os.scandir(path_testing) as it:
    for entry in it:
        if entry.name.endswith(".csv") and entry.is_file():
            #print(entry.name, entry.path)
            test_data = pd.read_csv(entry.path)
            # pre-processing input data, remove columns with more than half missing values
            test_data = test_data.dropna(thresh=len(test_data.index)/2, axis='columns')
            # pre-processing input data, remove columns with empty name
            test_data = test_data.loc[:, ~test_data.columns.str.contains('^Unnamed')]
            test_data = test_data.astype(str)   # convert all labels to str, mod 7/8rip()

            # pre-processing input data, remove columns with empty name
            test_data = test_data[test_data.columns.dropna()]  # drop col w/o no header
            test_data.columns = test_data.columns.str.strip()     # remove space b/a header

            # capitalize all the labels
            for i in range(len(test_data.iloc[-1,:])):
                s = test_data.iloc[-1,i]
                lst = [word[0].upper() + word[1:] for word in s.split()]
                test_data.iloc[-1,i] = " ".join(lst)
            # extract the label, which is the last row
            label = test_data.tail(1).to_dict('records')[0]

            # drop the label if it is exists. otherwise drop the last row
            test_data = test_data.drop(test_data.tail(1).index)
            # test_data = test_data.astype(str)
            # test_data = test_data.fillna("NA")
            # test_data.columns = test_data.columns.str.strip()

            loaded_model = pickle.load(open('matcher1.model', 'rb'))
            predicted_output = loaded_model.make_prediction(test_data)
            predicted_mapping = {}
            for k,v in predicted_output.items():
                predicted_mapping[k] = v[0]


            json_output = json.dumps(predicted_output, indent=4)
            filepath, filename = os.path.split(entry.path)
            print('File:', filename)
            print(json_output)
            # accuracy
            overlap = label.items() & predicted_mapping.items()
            label_ct = sum(1 for value in label.values() if value != 'Nan' )
            print("Accuracy for file \"" + filename + "\" is", f"{len(overlap)/label_ct:.1%}")
            # save json to output folder
            filePathNameWExt = path_output + filename + '.json'
            with open(filePathNameWExt, 'w', encoding='utf-8') as f:
                json.dump(predicted_output, f, ensure_ascii=False, indent=4)
