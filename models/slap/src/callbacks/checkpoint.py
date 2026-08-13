from lightning.pytorch.callbacks import ModelCheckpoint

class PTModelCheckpoint(ModelCheckpoint):
    FILE_EXTENSION = ".pt"
