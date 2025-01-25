pretrained_model = "/home4/khanhnd/cache/hub/models--utter-project--mHuBERT-147/snapshots/69467f5ee758b795c77abe33ac4a09dff1b82fdf"
# pretrained_model="/home4/khanhnd/cache/hub/models--facebook--wav2vec2-base/snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
# pretrained_model="/home4/khanhnd/self-KD/checkpoints/teacher_baseline/checkpoint-72000"

training_data = "/home4/khanhnd/self-condition/data/multi_lingual.tsv"
# training_data="/home4/khanhnd/self-condition/data/libri10h_phoneme.tsv"
vocab_char = "/home4/khanhnd/self-condition/vocab/multilingual.json"
# vocab_char="/home4/khanhnd/self-condition/vocab/vocab_char.json"
vocab_phoneme="/home4/khanhnd/self-condition/vocab/ipa.json"
# vocab_phoneme="/home4/khanhnd/self-condition/vocab/vocab_phoneme.json"
checkpoint_dir = "checkpoint_small/hubert_369_3_0.8_weight_0.7"
# training args
num_epochs = 80
batch_size = 2
cache_dir=f"cache"
learning_rate = 2e-5
print("Siuuuu")