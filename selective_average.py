import os 
import collections
import pandas as pd
import torch
from transformers import  HubertConfig,HubertForCTC
from model_hubert import HubertForCTCSelfCondPhoneme
import glob
from tqdm import tqdm
checkpoints_paths = glob.glob("/home4/khanhnd/self-condition/checkpoint_small/hubert_en_baseline_simple/chec*")
params_dict = collections.OrderedDict()
params_keys = None
num_models = 0

# for i, checkpoint in enumerate(checkpoint_wer.checkpoint.to_list()[:topk]):
for i, checkpoint in enumerate(checkpoints_paths):
    print(checkpoint)
    num_models += 1
    cpkt = HubertForCTCSelfCondPhoneme.from_pretrained(checkpoint)
    model_params = cpkt.state_dict()
    
    model_params_keys = list(model_params.keys())
    if params_keys is None:
        params_keys = model_params_keys
    elif params_keys != model_params_keys:
        raise KeyError(
            "Expected list of params: {}, "
            "but found: {}".format(params_keys, model_params_keys)
        ) 
    for k in params_keys:
        p = model_params[k]
        if isinstance(p, torch.HalfTensor):
            p = p.float()
        if k not in params_dict:
            params_dict[k] = p.clone().to(dtype=torch.float64)
            # NOTE: clone() is needed in case of p is a shared parameter
        else:
            params_dict[k] += p.to(dtype=torch.float64)

print('num_models:', num_models)
# num_models += 1

final_state_dict = collections.OrderedDict()

for k, v in tqdm(params_dict.items()):
    v.div_(num_models)
    # float32 overflow seems unlikely based on weights seen to date, but who knows
    float32_info = torch.finfo(torch.float32)
    for k, v in params_dict.items():
        v = v.clamp(float32_info.min, float32_info.max)
        final_state_dict[k] = v.to(dtype=torch.float32)

float32_info = torch.finfo(torch.float32)

config = HubertConfig.from_json_file(os.path.join(checkpoints_paths[-1], 'config.json'))

average_model = HubertForCTCSelfCondPhoneme(config=config)
average_model.load_state_dict(final_state_dict)


average_model.save_pretrained('/home4/khanhnd/self-condition/checkpoint_avg/hubert_en_baseline')