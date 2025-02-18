from multiprocessing import get_context
import torchaudio
from datasets import load_dataset
from jiwer import wer
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
)
import kenlm
from pyctcdecode import build_ctcdecoder
import torch
import numpy as np
import pandas as pd
import random
import os

device = "cuda"
final_checkpoint="nguyenvulebinh/wav2vec2-base-vietnamese-250h"
processor = Wav2Vec2Processor.from_pretrained(final_checkpoint)

model = Wav2Vec2ForCTC.from_pretrained(final_checkpoint).eval().to(device)

test_dataset = load_dataset(
    "csv",

    data_files="/home4/khanhnd/self-condition/data/multi_lingual_large.tsv",
    sep="\t",
    split="train",
    cache_dir="/home3/khanhnd/cache",
)
test_dataset=test_dataset.filter(lambda x: x["lan_id"]=="vi")


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

# test_dataset = test_dataset.select([i for i in range(60000,80000)])
test_dataset1 = test_dataset.map(speech_file_to_array_fn, num_proc=8)

from torch import nn
import time
import pandas as pd

# from normalize import normalizetmu

from torchmetrics.text import WordErrorRate

wer = WordErrorRate()



def map_to_pred(batch, pool):
    inputs = processor(
        batch["speech"], sampling_rate=16_000, padding=True, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    logits_list = logits.cpu().numpy()
    predicted_ids = torch.argmax(logits, dim=-1)
    best_text = processor.batch_decode(predicted_ids)
    batch["pred_text"] = best_text
    batch.pop("speech")

    batch.pop("sampling_rate")
    batch.pop("input_length")
    return batch

s = time.time()
with get_context("fork").Pool(processes=4) as pool:
    result = test_dataset1.map(
        map_to_pred, batched=True, batch_size=4, fn_kwargs={"pool": pool}
    )

result.set_format(
    type="pandas",
    columns=["audio_filepath", "pred_text", "transcript"],
)
# name = final_checkpoint.split("/")[-2]
name = "temp.csv"
save_path = f"result.csv"
result.to_csv(save_path, index=False, sep="|")
df = pd.read_csv(save_path, sep="|")
# df = df[["audio_path", "pred_text", "transcription", "score"]]

df = df.fillna("")
e = time.time()

with open("/home4/khanhnd/self-condition/result.txt", "a") as f:
    f.write(str(wer(list(df["pred_text"]), list(df["transcript"])))+"\n")
