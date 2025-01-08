# Auto-generated constants.py

ABBREVIATIONS = {
    "nproc_per_node": "nproc_per_node",
    
    "batch_size": "bs",
    "world_size": "world_size",
    "model": "model",
    "epochs": "e",
    "blr": "blr",
    "layer_decay": "layer_decay",
    "weight_decay": "weight_decay",
    "drop_path": "drop_path",
    
    
    "input_size": "input_size",
    'modality': 'mode',
    
    'balance': 'balance',
    'loss': 'loss',
    'use_sigmoid': 'sigmoid',
    'stats_source': 'sts',
    'more_augmentation': 'more_aug',
    'transform': 'trans'
}


def get_data_model(modality, experiment='scivias'):
    model_path = f"/data/core-kind/fatemeh/data/{modality}_weights.pth"
    if experiment == 'scivias':
        data_path = f"/data/core-kind/fatemeh/data/{modality}/organized/"
        nb_classes = 20
    else:
        nb_classes=5
        if modality == 'op':
            data_path = '/data/core-kind/fatemeh/data/IDRiD_data'
        else:
            data_path = '/data/core-kind/fatemeh/data/OCTID'
    return data_path, model_path, nb_classes