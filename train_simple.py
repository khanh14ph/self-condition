from typing import Dict, List, Optional, Union
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from datasets import load_dataset
from dataclasses import dataclass
from transformers import Trainer
from model import Wav2Vec2ForCTCSelfCond,Wav2Vec2ForCTCSelfCondBaseline
# from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2CTCTokenizer
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
import torch
import torchaudio
import os


import config

if os.path.exists("cache") == False:
    os.mkdir("cache")
print(config.cache_dir)
train_dataset = load_dataset(
    "csv",
    data_files=config.training_data,
    split="train",
    cache_dir=config.cache_dir,
    sep="\t",
)
# random_indices = [i for i in range(1000)]
# train_dataset = train_dataset.select(random_indices)
# train_dataset = train_dataset.filter(lambda example: example["is_groundtruth"] == True)
print("LEN: ", len(train_dataset))
chars_to_ignore_regex = (
    '[\,\̀\#\̃\_\̣\=\$\&\̉\?\̀\(\)\+\/\_\'"\{\}\<\>\|\`\~\*\&\^\@\.\!\́\-\;\:"\“\%\‘\”\�]'
)

tokenizer = Wav2Vec2CTCTokenizer(
    config.vocab_char,
    unk_token="<unk>",
    pad_token="<pad>",
    word_delimiter_token="|",
    bos_token="<s>",
    eos_token="</s>",
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["audio_filepath"])
    new_sr = 16000
    speech_array = torchaudio.functional.resample(
        speech_array, orig_freq=sampling_rate, new_freq=new_sr
    )
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = new_sr
    batch["target_text"] = (
        batch["transcript"] if batch["transcript"] is not None else "[UNK]"
    )
    batch["input_values"] = processor(
        [batch["speech"]], sampling_rate=batch["sampling_rate"]
    ).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor([batch["target_text"]]).input_ids[0]
    batch.pop("sampling_rate")
    batch.pop("speech")
    batch.pop("audio_filepath")

    # "speech","audio_path")
    return batch


func_suffix = ""

train_dataset = train_dataset.map(
    speech_file_to_array_fn,
    num_proc=8,
)
print(train_dataset)


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=True,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        # print(batch.keys())
        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# wer_metric = load_metric("wer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    label_str = ["im lặng" if label == "" else label for label in label_str]
    pred_str = ["im lặng" if pred == "" else pred for pred in pred_str]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# w2v_cfg=Wav2Vec2Config.from_pretrained("config.json")
# model=Wav2Vec2ForCTC(w2v_cfg)
model = Wav2Vec2ForCTCSelfCondBaseline.from_pretrained(
    config.pretrained_model,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.1,
    final_dropout=0.1,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    ctc_zero_infinity=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)
print("Model chosen:", type(model))

model.freeze_feature_extractor()
model.freeze_base_layers()
training_args = TrainingArguments(
    output_dir=config.checkpoint_dir,
    group_by_length=False,

    per_device_train_batch_size=config.batch_size,
    gradient_accumulation_steps=2,
    save_strategy="steps",
    logging_strategy="steps",
    num_train_epochs=config.num_epochs,
    ignore_data_skip=False,
    save_steps=2000,
    #   eval_steps=200000000000000000,
    logging_steps=10,
    save_safetensors=False,
    #   evaluation_strategy="epoch",
    dataloader_num_workers=6,
    learning_rate=config.learning_rate,
    warmup_steps=1000,
    #   label_smoothing_factor=0.1,
    save_total_limit=5,
    #   eval_accumulation_steps=1,
    report_to="tensorboard",
    #   load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    tokenizer=processor.feature_extractor,
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)
trainer.train()
# trainer.train(resume_from_checkpoint=True)
