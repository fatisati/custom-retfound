# Auto-generated defaults.py

defaults_dict = {
    "experiment": 'retfound',
    "nproc_per_node": 1,
    "master_port": 48798,
    "batch_size": 64,
    "world_size": 1,
    "model": "vit_large_patch16",
    "epochs": 300,
    "blr": "5e-4",
    "layer_decay": 0.65,
    "weight_decay": 0.05,
    "drop_path": 0.2,
    "nb_classes": 5,
    "data_path": "./IDRiD_data/",
    "finetune": "./RETFound_cfp_weights.pth",
    "input_size": 224,
    'modality': 'opt',
    'balance': 1,
    'loss': 'cross_entropy',
    'use_sigmoid': 0,
    'stats_source': 'imagenet',
    'more_augmentation': 0,
    'transform': 'retfound'
}
