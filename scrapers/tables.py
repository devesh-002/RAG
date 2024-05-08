import json
import os
COMPANY="AAPL"
def json_to_markdown(document,year,company):
    markdown_output = ""

# Iterate through each document
# for document in documents:
    markdown_output += f"# {year +' '+ company}\n\n"  # Assuming each document has a title
    # Iterate through each subtitle
    for subtitle, subtitle_ in document.items():
        markdown_output += f"## {subtitle}\n\n"
        # Iterate through each page
        try:
            for page,pages in document[subtitle].items():
                # Iterate through each item on the page
                for key,item in subtitle_[page].items():
                
                    if key=="text":
                        # If it's a text item, add it to markdown_output
                        markdown_output += item + "\n\n"
                    elif key=="tables":
                        # If it's a table item, convert it to markdown table
                        # print(item)
                        for _,table in item.items():
                            tables=table

                            if(tables==[]):continue
                            headers = tables[0]
                            markdown_output += "| " + " | ".join(headers) + " |\n"
                            # Add separator
                            markdown_output += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                            # Add rows
                            for row in tables[1:]:
                                markdown_output += "| " + " | ".join(row) + " |\n"
                            # Add newline after tables
                            markdown_output += "\n"
                        markdown_output += "\n"
                # Add newline between pages
                markdown_output += "\n"
        except:
            pass
    open("final_data_md_"+COMPANY+"/"+ year+"_"+COMPANY +".md", "w").write(markdown_output)

    
path="/home/devesh/projects/gatech/final_data_"+COMPANY+"/"
# file_path="/home/devesh/projects/gatech/final_data_MSFT/0001193125-09-158735.json"

# z=json_to_markdown(json.load(open(file_path, "r")), "2023", COMPANY)
# print(z)
# print(path)
for subdir, dirs, files in os.walk(path):
        for file in files:
            md=""
            test_name=""
            if(file.endswith(".json")):
                file_path=os.path.join(subdir, file)
                
                name=file.split("/")[-1].split("-")[1]
                name=int(name)
                print(name)
                if(name >=5):
                    # print(file_path)
                    data= open(file_path, "r").read()
                    # print(file.split("/")[-1])
                    test_name=str(name)
                    if(len(test_name)==1):
                        test_name="200"+test_name
                    else:
                        test_name="20"+test_name
                    json_to_markdown(json.load(open(file_path, "r")),test_name, COMPANY)

# open("final_data_"+COMPANY+".md", "w").write(out)
                     