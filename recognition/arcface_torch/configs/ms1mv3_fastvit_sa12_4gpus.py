"""
FastViT-SA12 fp16 training on 4 GPUs (A10)
"""
from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "fastvit_sa12"  # 10.9M params
config.resume = False
config.output = "work_dir/ms1mv3_arcface_fastvitSA12_4gpus_fp16"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 0.05
config.batch_size = 512
config.optimizer = "adamw"
config.lr = 0.001
config.verbose = 2000
config.dali = True

config.rec = "/home/ubuntu/data/MS1MV3_shuffled/"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 24
config.warmup_epoch = 0
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
