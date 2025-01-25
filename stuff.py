from viphoneme import vi2IPA
import eng_to_ipa as ipa
from tqdm import tqdm
import librosa
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
# df=pd.read_csv("/home4/khanhnd/self-condition/data/multi_lingual_large.tsv",sep="\t")
df=pd.read_csv("/home4/khanhnd/Ezspeech/data/vlsp2020_train.tsv",sep="\t")
df["lan_id"]=["vi" for i in df.iterrows()]
df1=pd.read_csv("/home4/khanhnd/self-condition/data/libri-train-clean-100.csv")
df1["audio_filepath"]=df1["path"]
df1["transcript"]=df1["transcription"]
df1["transcript"]=df1["transcript"].apply(lambda x:x.upper())
df1=df1[["audio_filepath","transcript"]]
df1["lan_id"]=["en" for i in df1.iterrows()]
# df["duration"]=df["audio_filepath"].progress_apply(lambda x: librosa.get_duration(path=x))
def to_ipa(row):
    if row["lan_id"]=="vi":
        res=vi2IPA(row["transcript"]).replace(".","").replace("_", " ").strip()
        return res
    if row["lan_id"]=="en":
        return ipa.convert(row["transcript"].lower())
# Apply with additional parameters

df['ipa_transcript'] = df.progress_apply(lambda row: to_ipa(row), axis=1)

for i in tqdm(df["ipa_transcript"]):
    if "ạ" in i:
        print(i)
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
with open('/home4/khanhnd/self-condition/vocab/ipa_large.json', 'w', encoding='utf-8') as json_file:
    json.dump(d, json_file, ensure_ascii=False, indent=4)
df.to_csv("/home4/khanhnd/self-condition/data/multi_lingual_large.tsv",sep="\t",index=False)
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