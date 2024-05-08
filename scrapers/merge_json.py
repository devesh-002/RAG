import json
import os 
COMPANY="MSFT"
path="/home/devesh/projects/gatech/final_data_"+COMPANY+"/"

final_dict={}
def iterate_over_files(dir_path):
    for subdir, dirs, files in os.walk(dir_path):
        file_json={}
        for file in files:
            if(file.endswith(".json")):
                file_path=os.path.join(subdir, file)
                # print(file_path)
                name=file_path.split("/")[-1].split("-")[1]
                name=int(name)
                if(name>=5):
                    # print(file_path)
                    # json=open(file_path, "r").read()
                    json_=json.load(open(file_path, "r"))
                    text=""
                    table=[]
                    for key in json_:
                        if(key!=None):
                            try:
                                for key2 in json_[key]:
                                    for key3 in json_[key][key2]:
                                        # if(key3=="date"):
                                        if(key3=="text"):
                                            text+=json_[key][key2][key3]
                                        else:
                                            table.append(json_[key][key2][key3])
                                            # print(json_[key][key2][key3])
                            except:
                                pass
                    test_name=str(name)
                    if(len(test_name)==1):
                        test_name="200"+test_name
                    else:
                        test_name="20"+test_name
                    new_json={ "text":text, "table":table}
                    new_json["company"]=COMPANY
                    new_json["year"]=test_name
                    
                    final_dict["data"]=new_json
                    

iterate_over_files(path)
with open("final_data_"+COMPANY+".json", "w") as fout:
    json.dump(final_dict , fout)
# json.dumps(final_dict, )
