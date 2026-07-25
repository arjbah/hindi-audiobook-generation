# Important Folders
* `laion_clap_training`
This folder is used to train LAION CLAP. Run `train_clap.py` to train.
* `voiceclap_training`
This folder is used for VoiceCLAP training. This is identical to the LAION CLAP code, just the audio encoder is swapped out. Run `train_clap.py` to train.
* `slap`
This folder is used to train SLAP. It is a modified version of the original Guinot SLAP repository. Run 
```bash
python src/train.py data=rasa model=slap "model/audio_encoder=htsat_audioset_slap" "model/text_encoder=muril_slap" trainer=hindi
```
to train.
* `mga_clap_training`
This folder is used to train MGA-CLAP. It is a modified version of the original MGA-CLAP repository. Simply run `pretrain.py` to train.

Note that the current code trains the CLAP model on the female and male subsets of Rasa Hindi. This is incorrect in practice because the generated audio would alternate between
sounding male and sounding female, which is bad for speaker consistency. However, fixing this is pretty simple.