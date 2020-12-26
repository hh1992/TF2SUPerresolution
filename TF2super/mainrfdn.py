import os
import matplotlib.pyplot as plt

from data import DIV2K
from model.edsr import edsr
from model.RFDNNET import rfdn
from train import RFDNTrainer


# Number of residual blocks
depth = 16

# Super-resolution factor
scale = 4

# Downgrade operator
downgrade = 'bicubic'

# Location of model weights (needed for demo)
weights_dir = f'weights/edsr-{depth}-x{scale}'
weights_file = os.path.join(weights_dir, 'weights.h5')

os.makedirs(weights_dir, exist_ok=True)

div2k_train = DIV2K(scale=scale, subset='train', downgrade=downgrade)
div2k_valid = DIV2K(scale=scale, subset='valid', downgrade=downgrade)

train_ds = div2k_train.dataset(batch_size=16, random_transform=True)
valid_ds = div2k_valid.dataset(batch_size=1, random_transform=False, repeat_count=1)
trainer = RFDNTrainer(model=rfdn(scale=scale),
                      checkpoint_dir=f'.ckpt/RFDNNET-{depth}-x{scale}')
# Train EDSR model for 300,000 steps and evaluate model
# every 1000 steps on the first 10 images of the DIV2K
# validation set. Save a checkpoint only if evaluation
# PSNR has improved.

trainer.train(train_ds,
              valid_ds.take(10),
              steps=300000,
              evaluate_every=1000,
              save_best_only=True)

# Restore from checkpoint with highest PSNR
trainer.restore()

# Evaluate model on full validation set
psnrv = trainer.evaluate(valid_ds)
print(f'PSNR = {psnrv.numpy():3f}')

# Save weights to separate location (needed for demo)
trainer.model.save_weights(weights_file)