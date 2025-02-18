from viphoneme import vi2IPA
import eng_to_ipa as ipa
from tqdm import tqdm
import librosa
import pandas as pd
tqdm.pandas()
import json
import librosa

df1=pd.read_csv("/home4/khanhnd/self-condition/data/valset_bilingual.tsv",sep="\t")   
df=df1.loc[df1["lan_id"]=="vi"]  
lst=list(df["transcript"])
text_lst=[]
for i in lst:
    text_lst+=i.split()
text_lst=list(set(text_lst))
d=json.load(open("ipa_map.json"))   
from dp.phonemizer import Phonemizer
phonemizer = Phonemizer.from_checkpoint('en_us_cmudict_ipa_forward.pt')
ipa_map=dict()
for i in tqdm(text_lst):
    
    if i in d.keys():
        ipa_map[i]=d[i] 
    else:
        ipa_form=vi2IPA(i).replace(".","").replace("_", " ").strip()
        if ipa_form!=i and ipa_form!="":
            ipa_map[i]=ipa_form 
        else:
            ipa_map[i]=phonemizer(i, lang='en_us')

ipa_map_english=dict()
df2=df1.loc[df1["lan_id"]=="en"] 
lst=list(df2["transcript"])
text_lst=[]
for i in lst:
    text_lst+=i.split()
text_lst=list(set(text_lst))
text_lst=[i.lower() for i in text_lst]
for i in tqdm(text_lst):
    ipa_form=ipa.convert(i.lower(),stress_marks=False)
    if "*" in ipa_form:
        ipa_form=   phonemizer(i.lower(), lang='en_us')
    ipa_map_english[i]=ipa_form 


    
def vi2ipa(x):
    x=x.split()
    x=[ipa_map[i] for i in x]   
    return " ".join(x)  
def en2ipa(x):
    x=x.split()
    x=[ipa_map_english[i] for i in x]   
    return " ".join(x)  
df=pd.concat([df,df2])
# df["duration"]=df["audio_filepath"].progress_apply(lambda x: librosa.get_duration(path=x))
def to_ipa(row):
    if row["lan_id"]=="vi":
        res=vi2ipa(row["transcript"])
        return res
    if row["lan_id"]=="en":
        return en2ipa(row["transcript"].lower())


df['ipa_transcript'] = df.progress_apply(lambda row: to_ipa(row), axis=1)

lst=list(df["transcript"])
s=""
for i in df["transcript"]:
    s+=i
s=list(set(s))
s=sorted(s)
s=[i for i in s if i!=" "]
d={
    "<pad>": 0,
    "<s>": 1,
    "</s>": 2,
    "<unk>": 3,
    "|": 4}
for idx,i in tqdm(enumerate(s),total=len(s)):
    d[i]=idx+5

import json
# with open('/home4/khanhnd/self-condition/vocab/multilingual.json', 'w', encoding='utf-8') as json_file:
#     json.dump(d, json_file, ensure_ascii=False, indent=4)


s=""
for i in df["ipa_transcript"]:
    s+=i
s=list(set(s))
s=sorted(s)
s=[i for i in s if i!=" "]
d={
    "<pad>": 0,
    "<s>": 1,
    "</s>": 2,
    "<unk>": 3,
    "|": 4}
for idx,i in tqdm(enumerate(s),total=len(s)):
    d[i]=idx+5
import json
# with open('vocab/ipa_large.json', 'w', encoding='utf-8') as json_file:
#     json.dump(d, json_file, ensure_ascii=False, indent=4)
df.to_csv("/home4/khanhnd/self-condition/data/valset_bilingual.tsv",sep="\t",index=False)
# df=pd.read_csv("/home4/khanhnd/self-condition/data/multi_lingual_large.tsv",sep="\t")
# check_lst=[]

# import json
# lst=set([i for i in json.load(open("/home4/khanhnd/self-condition/vocab/ipa.json"))])

# for idx, i in tqdm(df.iterrows(),total=len(df)):
#     try:
#         check=False
#         temp = list(i["ipa_transcript"].replace(" ", "|"))
#         for j in temp:
#             if j not in lst:
#                 print(i["transcript"])
#                 break
#         else:
#             check=True
        
#     except:
#         check=False
#         print(i["transcript"])
#     check_lst.append(check)
# df["check"]=check_lst
# df1=df.loc[df["check"]==True]

# print(len(df))
# df1.to_csv("/home4/khanhnd/self-condition/data/multi_lingual_large.tsv",sep="\t",index=False)