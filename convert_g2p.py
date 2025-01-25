from g2p_en import G2p

texts = ["I HAVE $250 in my pocket.", # number -> spell-out
         "popular pets, e.g. cats and dogs", # e.g. -> for example
         "I refuse to collect the refuse around here.", # homograph
         "I'm an activationist."] # newly coined word
g2p = G2p()
import pandas as pd
df=pd.read_csv("/home4/khanhnd/self-condition/data/libri10h_phoneme.tsv",sep="\t")
lst=[]
from tqdm import tqdm
for idx,i in tqdm(df.iterrows(),total=len(df)):
    text= i["transcript"]
    out = g2p(text)
    out=["_"+j  if j!=" " else " " for j in out]
    out="".join(out)
    lst.append(out)
df["phoneme_transcript"]=lst
# lst=sorted(list(set(lst)))
# with open("phoneme_vocab.txt","w") as f:
#     for i in lst:
#         f.write(i+"\n")
df.to_csv("/home4/khanhnd/self-condition/data/libri10h_phoneme.tsv",sep="\t",index=False)