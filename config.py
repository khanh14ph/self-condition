pretrained_model = "/home4/khanhnd/cache/hub/models--utter-project--mHuBERT-147/snapshots/69467f5ee758b795c77abe33ac4a09dff1b82fdf"
# # pretrained_model="/home4/khanhnd/cache/hub/models--facebook--hubert-base-ls960/snapshots/dba3bb02fda4248b6e082697eee756de8fe8aa8a"
# pretrained_model="/home4/khanhnd/self-condition/checkpoint_small/hubert_baseline/checkpoint-240000"

training_data = "/home4/khanhnd/self-condition/data/multilingual_large.tsv"
val_data="data/valset_bilingual.tsv"
# training_data="/home4/khanhnd/self-condition/data/libri10h_phoneme.tsv"
vocab_char = "/home4/khanhnd/self-condition/vocab/multilingual.json"
# vocab_char="/home4/khanhnd/self-condition/vocab/vocab_char.json"
vocab_phoneme="/home4/khanhnd/self-condition/vocab/ipa_large.json"
# vocab_phoneme="/home4/khanhnd/self-condition/vocab/lid.json"
checkpoint_dir = "/home4/khanhnd/self-condition/checkpoint_small/conformer_vi_2,4,6,8,10_2,4,6,8,10"
# training args
num_epochs = 30
batch_size = 4
cache_dir=f"cache"
learning_rate = 2e-5
print("Siuuuu")