"""
FastViT-SA12 fp16 training on 2 GPUs (GTX 1080 Ti)
"""
from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "fastvit_sa12"  # 10.9M params
config.resume = False
config.output = "/mnt/trainingdb0/data/face-recognition/checkpoints/arcface_torch_models/ms1mv3_arcface_fastvitSA12_2gpus_fp16"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 350
config.lr = 0.04
config.verbose = 3000
config.dali = False

config.rec = "/mnt/trainingdb0/data/face-recognition/ms1m-retinaface-t1/"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 24
config.warmup_epoch = 0
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
