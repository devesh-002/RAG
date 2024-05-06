import requests
from bs4 import BeautifulSoup
import re
from unidecode import unidecode 
import json
# https://gist.github.com/anshoomehra/ead8925ea291e233a5aa2dcaa2dc61b2
import pandas as pd
import os

TICKER="MSFT"
path="/home/devesh/projects/gatech/sec-edgar-filings/MSFT/10-K/0001193125-13-310206/full-submission.txt"

def find_tables(item):
  
    # item=unidecode(item)
    item=item.lower()
    soup=BeautifulSoup(item, 'lxml')
  
    tables=soup.find_all('table')
    if(len(tables)==0):
        return []
    total_tab=[]
    
    for table in tables:
        
        final_table=[]
        for row in table.find_all("tr"):
            cell_list=[]
            for cell in row.find_all(["td", "th"]):
                cell_text = cell.text.strip()
                k = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.%- "
                getVals = list(filter(lambda x: x in k, cell_text))
                cell_text = "".join(getVals)
                if(cell_text=="" or cell_text==" " or cell_text=="\n" or cell_text=="$"):
                    continue
            
                cell_list.append(cell_text)
            if(cell_list==[]): continue
            final_table.append(cell_list)
        
        
        
        if(final_table==[] or final_table==None): continue
        try:
         if((len(final_table[0])+len(final_table[1]))==len(final_table[2])):
            final_table[0].extend(final_table[1])
            del final_table[1]
            pass
        except :
            continue
        if((len(final_table[0])+1)==len(final_table[1])): 
           final_table[0].insert(0, "Item")
            
        indexes=[]
        for i in range(0,len(final_table)):
            if(len(final_table[i])==1):
                indexes.append(i)
        
        indexes=sorted(indexes, reverse=True)
 
        for i in indexes:
            del final_table[i]
        total_tab.append(final_table)
    
    return total_tab
    
def get_item(item):
    item=item.lower()
    # print(item)
    item=' '.join(item.splitlines())
    item=re.sub(r"<table*.*?<\/table>", ' ', item)
    # item=re.sub(r"(>ITEM(\s|&#160;|&nbsp;)(1|2|6|7A|7|8|9)\.{0,1})", ' ', item)
    # print(item)
    soup=BeautifulSoup(item, 'lxml')
    txt= soup.get_text()
    # print(soup)
    if(len(txt)<=100):
        return ""
    return txt
    

def item_preprocessing(item):
    item=item.replace("&#151;", "-")
    soup=BeautifulSoup(item, 'lxml')
    html=""
    lenn=len(soup.find_all("p",attrs={"style":'page-break-before:always'}))
    print(lenn)
    if(lenn==0):
        return str(item)
    x=[]
    print(lenn)
    
    for tag in soup.find("p",attrs={"style":'page-break-before:always'}).next_siblings:
        # print("lol")
        if  tag.name == "p" and tag.get("style")=="page-break-before:always":
            x.append(html)    
            html=""
            
        else:
            html+=(str(tag))

    return x
    
def parser(raw_10k,name="test"):
    # try:
        doc_start_pattern = re.compile(r'<DOCUMENT>')
        doc_end_pattern = re.compile(r'</DOCUMENT>')
        type_pattern = re.compile(r'<TYPE>[^\n]+')

        doc_start_is = [x.end() for x in doc_start_pattern.finditer(raw_10k)]
        doc_end_is = [x.start() for x in doc_end_pattern.finditer(raw_10k)]

        doc_types = [x[len('<TYPE>'):] for x in type_pattern.findall(raw_10k)]

        document = {}
        for doc_type, doc_start, doc_end in zip(doc_types, doc_start_is, doc_end_is):
            if doc_type == '10-K':
                document[doc_type] = raw_10k[doc_start:doc_end]
        document['10-K'][0:500]
        # Write the regex
        
        # FOR AAPL REGEX:  (ITEM\s(1|2|6|7A|7|8|9))|
        regex = re.compile(r"(>ITEM(\s|&#160;|&nbsp;)(1A|2|6|7A|7|8|9)\.{1})")
        
        # FOR MSFT REGEX: 
        # regex = re.compile(r'(>ITEM(\s|&#160;|&nbsp;)(1A|1B|6|7A|7|8|9)\.{0,1})|(ITEM\s(1A|1B|7A|7|8))')

        # Use finditer to math the regex
        matches = regex.finditer(document['10-K'])
        # print("YPP")
        for match in matches:
            print(match)

        # Matches
        matches = regex.finditer(document['10-K'])

        # Create the dataframe
        test_df = pd.DataFrame([(x.group(), x.start(), x.end()) for x in matches])

        test_df.columns = ['item', 'start', 'end']
        test_df['item'] = test_df.item.str.lower()
        test_df.replace('&#160;',' ',regex=True,inplace=True)
        test_df.replace('&nbsp;',' ',regex=True,inplace=True)
        test_df.replace(' ','',regex=True,inplace=True)
        test_df.replace('\.','',regex=True,inplace=True)
        test_df.replace('>','',regex=True,inplace=True)

        # Drop duplicates
        pos_dat = test_df.sort_values('start', ascending=True).drop_duplicates(subset=['item'], keep='last')

        # Display the dataframe
        # Set item as the dataframe index
        pos_dat.set_index('item', inplace=True)
        
        item_1a_raw = document['10-K'][pos_dat['start'].loc['item1a']:pos_dat['start'].loc['item2']]
        item_6_raw = document['10-K'][pos_dat['start'].loc['item6']:pos_dat['start'].loc['item7']]
        item_9_raw = document['10-K'][pos_dat['start'].loc['item9']:]

        # Get Item 7
        item_7_raw = document['10-K'][pos_dat['start'].loc['item7']:pos_dat['start'].loc['item8']]

        # Get Item 7a
        item_7a_raw = document['10-K'][pos_dat['start'].loc['item7a']:pos_dat['start'].loc['item8']]
        # print(item_7a_raw)
        item_8_raw = document['10-K'][pos_dat['start'].loc['item8']:pos_dat['start'].loc['item9']]
        # item_1a_content = BeautifulSoup(item_8_raw, 'lxml')
        # item_7a_content = BeautifulSoup(item_6_raw, 'lxml')
        # hrs = item_1a_content.find_all('hr')
        # print(item_8_raw)
        final_dict = {"Item 1A":None, "Item 6":None, "Item 7":None,  "Item 8":None}
        item_dict = {item_1a_raw:"Item 1A", item_6_raw:"Item 6", item_7_raw:"Item 7", item_7a_raw:"Item 7A", item_8_raw:"Item 8"}
        for item in [item_1a_raw,  item_7_raw, item_8_raw]:
            
            preprocess=BeautifulSoup(item, 'lxml')
            check=preprocess.find_all("hr")
            if(len(check)>0):
                preprocess=item_preprocessing(item)
            else:
                preprocess=[item]
            print(item_dict[item])
            x=0
            pages={}
            for idx,pre in enumerate(preprocess):
                if(pre=="" or pre=="\n" or pre==None): continue
                tablesxx=find_tables(pre)
                text=get_item(pre)
                tables={}
                if(tables==[] and text==""): continue
                for idxx,table in enumerate(tablesxx):
                    tables[idxx]=table
                    # print(table)
                    pass
                print("PAGE NUMBER: ", x);x+=1
                pages[idx]={"tables":tables, "text":text}
                print(text)
            final_dict[item_dict[item]]=pages
        json.dump(final_dict, open("final_data_"+TICKER+"/"+name+".json", "w"))
    # except:
    #     print("ERRRRRRRR")
    #     pass
        
def iterate_over_files(dir_path):
    for subdir, dirs, files in os.walk(dir_path):
        for file in files:
            if(file.endswith(".txt")):
                file_path=os.path.join(subdir, file)
                
                name=subdir.split("/")[-1].split("-")[1]
                name=int(name)
                if(name<=5 and name >=5):
                    # print(file_path)
                    data= open(file_path, "r").read()
                    print(subdir.split("/")[-1])
                    parser(data,subdir.split("/")[-1])  
raw_10k = open(path, "r").read()
# parser(raw_10k)
iterate_over_files("/home/devesh/projects/gatech/sec-edgar-filings/MSFT/10-K")