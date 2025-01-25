lst=open("/home4/khanhnd/asr-phoneme/vocab/lexicon_raw.txt").read().split("\n")[:-1 ]
map_dict={i.split("\t")[0]:"".join(i.split("\t")[1].split()) for i in lst}
def sentence2phoneme(x):
    try:
        x_lst=x.split()
        y_lst=[map_dict[i] for i in x_lst]
        return " ".join(y_lst)
    except:
        return "UNK"
import pandas as pd
df=pd.read_csv("/home4/khanhnd/vivos/train.tsv",sep="\t")
print(len(df))
df["phoneme_transcript"]=df["transcript"].apply(sentence2phoneme)
df=df.loc[df["phoneme_transcript"]!="UNK"]
print(len(df))
df.to_csv("train.tsv",sep="\t",index=False)