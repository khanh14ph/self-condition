from utils.text.text_normalization_tuannd import preprocess
import numpy as np

from datasets import load_dataset, concatenate_datasets
from evaluate import load
from dataclasses import dataclass

import os
# from transformers import Trainer
from custom_trainer import TrainerWithAlpha
from transformers import TrainingArguments
from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
)
from torch.utils.data import DataLoader
from transformers import get_scheduler
# from models import ConformerCTC, ConformerCTCMoE
# from models.conformer_espnet_lid import HubertForCTC
from models.e_branchformer_espnet_lid import HubertForCTC
import torch
from typing import Dict, List, Optional, Union
import configs
import warnings
import torchaudio
from tqdm import tqdm
from transformers import set_seed
set_seed(1412)

warnings.filterwarnings('ignore')
# LOAD MODEL
model_id = 'utter-project/mHuBERT-147'
cache_dir = "/home4/vuhl/asr/cache"
print("Save dir: ", configs.save_dir)
# model_id = '/home4/vuhl/asr/asr-experiments/checkpoints/multilingual/mhubert/libri_clean_100_cross_att_3/checkpoint-132210'

print("Model loaded from: ", model_id)
tokenizer = Wav2Vec2CTCTokenizer(
    configs.vi_vocab_file, 
    unk_token='[UNK]', 
    pad_token='[PAD]', 
    word_delimiter_token='|'
)
lid_tokenizer = Wav2Vec2CTCTokenizer(configs.lid_vocab, pad_token="[PAD]")
alpha = configs.alpha
beta = configs.beta
print("Initial alpha: ", alpha)
print("Initial beta: ", beta)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
print(f'is fast {processor.tokenizer.is_fast}')
print(f'Vocab size {len(processor.tokenizer) - 2}')

model = HubertForCTC.from_pretrained(
    model_id,
    attention_dropout=0.1,
    apply_spec_augment=True,
    activation_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.1,
    mask_time_prob=0.05,
    layerdrop=0.1,
    final_dropout=0.3,
    ctc_loss_reduction='mean',
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)-2, # exclude <eos> and <bos>
    ignore_mismatched_sizes=True,
    cache_dir=cache_dir,
)
# model.freeze_feature_encoder()
model.freeze_base_model()
model.hubert.eval() # turn off dropout

print(f'Total parameters {(sum(p.numel() for p in model.parameters()))/1e6:.2f}M')
print(f"Total trainable parameters: {(sum(p.numel() for p in model.parameters() if p.requires_grad))/1e6:.2f}M")

en_train_csv = ['/home4/tuannd/vbee-asr/self-condition-asr/csv/libri-train-clean-100.csv']

# vi_eval_csv = ['/home4/vuhl/asr/multilingual-asr/csvs/fleurs-vi_vn-valid.csv']
en_eval_csv = ['/home4/tuannd/vbee-asr/self-condition-asr/csv/libri-dev-clean.csv']

en_test_csv = ['/home4/tuannd/vbee-asr/self-condition-asr/csv/libri-test-clean.csv']

en_train_dataset = load_dataset("csv", data_files=en_train_csv, split='train', cache_dir=os.path.join(cache_dir, "train"))

en_valid_dataset = load_dataset("csv", data_files=en_eval_csv, split='train', cache_dir=os.path.join(cache_dir, "valid"))

en_test_dataset = load_dataset("csv", data_files=en_test_csv, split='train', cache_dir=os.path.join(cache_dir, "test"))

def fix_split(batch, split):
    batch['path'] = [f"{split}/{path}" for path in batch['path']]

def preprocess_text(sample, iso_code):
    sample["transcription"] = preprocess(sample["transcription"], iso_code=iso_code, remove_numbers=False, remove_brackets=True)
    if iso_code == "vi":
        sample["lid_labels_utt"] = torch.Tensor([0])
    elif iso_code == "en":
        sample["lid_labels_utt"] = torch.Tensor([1])
    return sample

def add_lid_word(batch, lang):
    """
    Convert each word in the transcription into a LID token
    LID word level
    """
    lid_token_id = None
    if lang == "vi":
        lid_token_id = "[VI]"
    elif lang == "en":
        lid_token_id = "[EN]"
    assert lid_token_id != None, "No language ID passed, or the passed language ID is invalid. Expected one of the following: ['vi', 'en']."

    batch['lid_token'] = (lid_token_id + " ") * len(batch['transcription'].split(" "))  # for LID at word level
    batch['lid_token'] = batch['lid_token'].strip()

    return batch
    
def speech_file_to_array_fn(sample):
    # sample['speech'] = sample['audio']['array']
    audio, sample_rate = torchaudio.load(sample['path'])
    # print(f'sample rate {_}')
    audio = torchaudio.functional.resample(audio, sample_rate, 16000)
    sample['speech'] = audio[0].numpy()
    sample['target_text'] = sample['transcription']
    sample['target_lid'] = sample['lid_token']
    sample['num_samples'] = len(sample['speech'])

    return sample

def prepare_dataset(batch): 
    batch['input_values'] = processor(batch['speech'], sampling_rate=16_000).input_values
    with processor.as_target_processor():
        batch['labels'] = processor(batch['target_text']).input_ids
    
    batch['lid_labels'] = lid_tokenizer(batch['target_lid']).input_ids

    return batch

def get_num_frames(sample):
    sample['num_frames'] = model._get_feat_extract_output_lengths(sample['num_samples'])
    sample['ilen'] = sample['num_frames']

    return sample


# # sampling for sanity checking
# en_train_dataset = en_train_dataset.select(range(1))
# en_valid_dataset = en_valid_dataset.select(range(1))
# en_test_dataset = en_test_dataset.select(range(1))


# vi_train_dataset = vi_train_dataset.map(preprocess_text, fn_kwargs={"iso_code": "vi"})
en_train_dataset = en_train_dataset.map(preprocess_text, fn_kwargs={"iso_code": "en"})

# vi_train_dataset = vi_train_dataset.map(add_lid_word, fn_kwargs={"lang": "vi"})
en_train_dataset = en_train_dataset.map(add_lid_word, fn_kwargs={"lang": "en"})


# vi_valid_dataset = vi_valid_dataset.map(preprocess_text, fn_kwargs={"iso_code": "vi"})
en_valid_dataset = en_valid_dataset.map(preprocess_text, fn_kwargs={"iso_code": "en"})


# vi_valid_dataset = vi_valid_dataset.map(add_lid_word, fn_kwargs={"lang": "vi"})
en_valid_dataset = en_valid_dataset.map(add_lid_word, fn_kwargs={"lang": "en"})

# vi_test_dataset = vi_test_dataset.map(preprocess_text, fn_kwargs={"iso_code": "vi"})
en_test_dataset = en_test_dataset.map(preprocess_text, fn_kwargs={"iso_code": "en"})


# vi_test_dataset = vi_test_dataset.map(add_lid_word, fn_kwargs={"lang": "vi"})
en_test_dataset = en_test_dataset.map(add_lid_word, fn_kwargs={"lang": "en"})


# train_dataset = concatenate_datasets([vi_train_dataset, en_train_dataset])
# valid_dataset = concatenate_datasets([vi_valid_dataset, en_valid_dataset])
# test_dataset = concatenate_datasets([vi_test_dataset, en_test_dataset])
train_dataset = en_train_dataset
valid_dataset = en_valid_dataset
test_dataset = en_test_dataset

# # shuffle
# train_dataset = train_dataset.shuffle(seed=42)
# valid_dataset = valid_dataset.shuffle(seed=42)
# test_dataset = test_dataset.shuffle(seed=42)

# # sampling for sanity checking
# train_dataset = train_dataset.select(range(1))
# valid_dataset = valid_dataset.select(range(1))
# test_dataset = test_dataset.select(range(1))


print(train_dataset[0]['path'])
print(train_dataset[0]['transcription'])
print(train_dataset[0]['lid_token'])
print(train_dataset[0]['lid_labels_utt'])


train_dataset = train_dataset.map(speech_file_to_array_fn)
valid_dataset = valid_dataset.map(speech_file_to_array_fn)
test_dataset = test_dataset.map(speech_file_to_array_fn)

# filter
train_dataset = train_dataset.filter(lambda x: x['num_samples'] <= 240000)
valid_dataset = valid_dataset.filter(lambda x: x['num_samples'] <= 240000)
test_dataset = test_dataset.filter(lambda x: x['num_samples'] <= 240000)

train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names.remove('num_samples'), batch_size=40, num_proc=40, batched=True)
valid_dataset = valid_dataset.map(prepare_dataset, remove_columns=valid_dataset.column_names.remove('num_samples'), batch_size=40, num_proc=40, batched=True)
test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names.remove('num_samples'), batch_size=40, num_proc=40, batched=True)

train_dataset = train_dataset.map(get_num_frames, remove_columns=['path', 'transcription', 'speech', 'target_text', 'target_lid', 'num_samples'], num_proc=40)
valid_dataset = valid_dataset.map(get_num_frames, remove_columns=['path', 'transcription', 'speech', 'target_text', 'target_lid', 'num_samples'], num_proc=40)
test_dataset = test_dataset.map(get_num_frames, remove_columns=['path', 'transcription', 'speech', 'target_text', 'target_lid', 'num_samples'], num_proc=40)


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    lid_tokenizer: Wav2Vec2CTCTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        # print(f'number of feature {len(features)}')
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        lid_label_features = [{"input_ids": feature["lid_labels"]} for feature in features]

        # ilen = [{"ilen": feature["ilen"]} for feature in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        lid_labels_batch = self.lid_tokenizer.pad(
                lid_label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        lid_labels = lid_labels_batch["input_ids"].masked_fill(lid_labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        batch["lid_labels"] = lid_labels
        # batch["lid_labels_utt"] = lid_label_utt_features

        if 'ilen' in features[0]:
            batch["ilen"] = [feature["ilen"] for feature in features]
            # batch["ilen"] = torch.Tensor(batch["ilen"])

        batch['lid_labels_utt'] = torch.Tensor([feature['lid_labels_utt'] for feature in features])
        batch['lid_labels_utt'] = batch['lid_labels_utt'].view(-1).type(torch.int64)

        batch['alpha'] = torch.Tensor([alpha])
        batch['beta'] = torch.Tensor([beta])

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True, lid_tokenizer=lid_tokenizer)

wer_metric = load('wer')

def compute_metrics(pred):
    """
    Used with HuggingFace Trainer.
    """
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    transcript_label_ids = pred.label_ids[0]
    transcript_label_ids[transcript_label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(transcript_label_ids, group_tokens=False)
    print(f'pred_string {pred_str} | label {label_str}')
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

training_args = TrainingArguments(
    output_dir=configs.save_dir,
    per_device_train_batch_size=configs.per_device_train_batch_size,
    gradient_accumulation_steps=configs.gradient_accumulation_steps,
    logging_strategy='steps',
    logging_steps=100,
    evaluation_strategy='epoch',
    # eval_steps=2000,
    num_train_epochs=configs.num_epochs,
    ignore_data_skip=False,
    fp16=False,
    save_strategy='epoch',
    save_total_limit=configs.save_total_limit,
    dataloader_num_workers=8,
    learning_rate=configs.learning_rate,
    warmup_ratio=configs.warmup_ratio,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-8,
    report_to='tensorboard',
    lr_scheduler_type=configs.lr_scheduler_type,
    # max_grad_norm=100.0,
    save_safetensors=False,
    load_best_model_at_end= True,
)

trainer = TrainerWithAlpha(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    # eval_dataset=test_dataset,
    # eval_dataset=train_dataset,
    tokenizer=processor.feature_extractor,
)

with torch.autograd.set_detect_anomaly(True):
    trainer.train()
    # trainer.train(resume_from_checkpoint=True)  
    # print(trainer.evaluate())

