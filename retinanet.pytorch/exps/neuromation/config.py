import os

train_path = '/var/hackaton/object_detection/set/SynteticSet20180323/data'
val_path = '/var/hackaton/object_detection/val/SynteticSet20180323_val/data'

backbone = 'resnet101'
classes = ['4601201018007', '4600494000393', '5449000050939', '5449000044709', 'noise_bottle', '4601201001412', '4601201018038', '4600494513893']
mean, std = (60.34319153, 68.21011999, 54.10308942), (6.28966094e-27, 5.37032107e-27, 5.30342856e-27)
scale = None

batch_size = 2
lr = 0.01
momentum = 0.9
weight_decay = 1e-4
num_epochs = 124
lr_decay_epochs = [83, 110]
num_workers = 2

eval_while_training = False
eval_every = 10
