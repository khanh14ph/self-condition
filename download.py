# from datasets import load_dataset
# import torch
# import torchaudio
# from tqdm import tqdm
# fleurs = load_dataset("google/fleurs", "vi_vn", split="test")
# with open("data/fleur_test_vi.tsv","w") as f:
#     f.write("audio_filepath\ttranscript\n")
#     for i in tqdm(fleurs):
#         path="/home4/khanhnd/self-condition/audio_fleur/"+i["audio"]["path"].split("/")[-1]
#         arr=torch.tensor(i["audio"]["array"]).reshape(1,-1)
#         torchaudio.save(path,arr,sample_rate=16000)
#         f.write(path+"\t"+i["transcription"]+"\n")
import torch
a=torch.load("/home4/khanhnd/self-condition/checkpoint_small/hubert_2,4,6,8,10/checkpoint-314490/pytorch_model.bin")
print(a["hubert.feature_projection.projection.bias"][0])