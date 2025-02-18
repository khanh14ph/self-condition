from multiprocessing import get_context
import torchaudio
from datasets import load_dataset
from jiwer import wer
from transformers import (
    # Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    HubertForCTC
)
from model import Wav2Vec2ForCTCSelfCond,Wav2Vec2ForCTCSelfCondBaseline,Wav2Vec2ForCTCSelfCondPhoneme

from model_hubert import HubertForCTCSelfCondPhoneme
# from model_phoneme import Wav2Vec2ForCTC
import kenlm
from pyctcdecode import build_ctcdecoder
import torch
import numpy as np
import pandas as pd
import random
import os
      
device = "cuda"
from glob import glob
final_checkpoint="/home4/khanhnd/self-condition/checkpoint_small/conformer_en_baseline/checkpoint-2000"
model = HubertForCTCSelfCondPhoneme.from_pretrained(final_checkpoint).eval().to(device)
vocab_file = "/home4/khanhnd/self-condition/vocab/multilingual.json"
# vocab_file="/home4/khanhnd/self-condition/vocab/ipa.json"
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file,
    # "/home4/khanhnd/thesis/asr_base/vocab.json" ,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    word_delimiter_token="|",
    replace_word_delimiter_char=" ",
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
import glob

import json

# Opening JSON file
with open(vocab_file) as json_file:

    my_dict = json.load(json_file)

my_dict = processor.tokenizer.get_vocab()
labels = sorted(my_dict, key=lambda k: my_dict[k]) 
test_dataset = load_dataset(
    "csv",
    # data_files="/home4/khanhnd/self-condition/data/fleur_test_vi.tsv",
    data_files="/home4/khanhnd/vivos/test.tsv",
    # data_files="/home4/khanhnd/self-condition/VLSP-ASR2020-testset/vlsp2020-test-task-1.tsv",
    # data_files="/home4/khanhnd/self-condition/data/libri-test-clean.tsv",
    sep="\t",
    split="train",
    cache_dir="/home3/khanhnd/cache",
) 


decoder = build_ctcdecoder(
    labels,
    kenlm_model_path="/home4/khanhnd/self-condition/data/corpus.arpa",  # either .arpa or .bin file
    alpha=0.5,  # tuned on a val set
    beta=1.0,  # tuned on a val set
)
# random_indices = [i for i in range(2500)]
# test_dataset = test_dataset.select(random_indices)


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["audio_filepath"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["input_length"] = len(batch["speech"]) / sampling_rate
    if batch["transcript"] == None:

        batch["transcript"] = ""
    else:
        batch["transcript"] = batch["transcript"]

    return batch

test_dataset1 = test_dataset.map(speech_file_to_array_fn, num_proc=8)


import pandas as pd




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(count_parameters(model))
def map_to_pred(batch, pool):
    inputs = processor(
        batch["speech"], sampling_rate=16_000, padding=True, return_tensors="pt"
    )
    length_lst=[len(i)/16000 for i in batch["speech"]]
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits= model(**inputs).logits
    logits_list = logits.cpu().numpy()

    res = decoder.decode_beams_batch(
        logits_list=logits_list, pool=pool, beam_width=50
    )
    best_text = [i[0][0] for i in res]
    
    predicted_ids = torch.argmax(logits, dim=-1)
    greedy_best_text = processor.batch_decode(predicted_ids)
    print(best_text)
    batch["pred_text"] = best_text
    # print(greedy_best_text)
    batch["greedy_pred_text"] = greedy_best_text
    batch.pop("speech")

    batch.pop("sampling_rate")
    batch.pop("input_length")
    return batch

import time
start=time.time()
with get_context("fork").Pool(processes=16) as pool:
    result = test_dataset1.map(
        map_to_pred, batched=True, batch_size=1, fn_kwargs={"pool": pool}
    )
result.set_format(
    type="pandas",
    columns=["audio_filepath", "pred_text","greedy_pred_text", "transcript"],
)

# name = final_checkpoint.split("/")[-2]
save_path = f"result/temp.tsv"
result.to_csv(save_path, index=False, sep="\t")



df = pd.read_csv(save_path, sep="\t")
# df = df[["audio_path", "pred_text", "transcription", "score"]]

df = df.fillna("")
def get_wer(x,y):
    
    s=[]
    for i,j in zip(x,y):
        i=i.replace("\"","")
        j=j.replace("\"","")
        if i=="":
            i="<unk>"
        if j=="":
            j="<unk>"
        s.append(wer(i,j))
    return sum(s)/len(s)

print(get_wer(list(df["transcript"]), list(df["pred_text"])))

print(get_wer(list(df["transcript"]), list(df["greedy_pred_text"])))