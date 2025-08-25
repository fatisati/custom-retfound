from util.datasets import *
import torch
import models_vit
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from sklearn.metrics import classification_report
from auto_sbatch.constants import *
from auto_sbatch.name import *
from auto_sbatch.defaults import *
from my_functions import *
import pickle as pkl

class Trainer:
    def __init__(self, device='cpu', **kwargs):

        # init args
        # get defaults
        # update defaults with kwargs
        # update args dict with new params
        # continue with the rest of the code
        print('new trainer')
        self.args = Args()
        self.params = defaults_dict.copy()
        self.params.update(kwargs)
        self.args.__dict__.update(self.params)

        self.data_path, _, nb_classes = get_data_model(
            self.params["modality"], self.params["experiment"]
        )
        self.args.data_path =  self.data_path
        self.args.nb_classes = nb_classes
        self.model_name = generate_model_name(defaults_dict, self.params, ABBREVIATIONS)
        self.out_dir = f"./out_files/{self.model_name}"
        self.args.data_path = self.data_path
        self.model_path = f"{self.out_dir}/checkpoint-best.pth"
        self.device = device
        
        set_mean_std(self.args)
        self.model = None

    def get_args(self):
        args = Args()
        args.data_path = self.data_path
        return args

    def get_data(self, split):
        dataset = build_dataset(is_train=split, args=self.args)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            drop_last=True,
        )
        return data_loader

    def get_model(self, args):
        return models_vit.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )

    def load_model(self, model_path):
        print('loading model...')
        model = self.get_model(self.args)
        checkpoint = torch.load(model_path, map_location="cpu")

        print("Load pre-trained checkpoint from: %s" % model_path)

        model.load_state_dict(checkpoint["model"])
        model.to(self.device)
        return model

    def get_model_out(self, split):
        # report = self.load_item(self.out_dir, split)
        # if report is not None:
        #     all_labels = self.load_item(self.out_dir, split, 'labels')
        #     all_preds = self.load_item(self.out_dir, split, 'preds')
        #     cm = confusion_matrix(all_labels, all_preds)
        #     print('report exist, loaded')
        #     return cm, report, all_labels, all_preds
        
        if self.model is None:
            self.model = self.load_model(self.model_path)
        
        data_loader = self.get_data(split)
        all_preds = []
        all_labels = []

        # Iterate over the data loader
        for data in tqdm(data_loader, desc=f"Getting model output for {split}, {self.model_path}"):
            images, labels = data

            # Move images and labels to the same device as the model
            images = images.to(self.device)  # Ensure 'self.device' is defined
            labels = labels.to(self.device)

            # Get model predictions
            outputs = self.model(images)
            preds = torch.argmax(
                outputs, dim=1
            )  # Get the class with the highest probability

            # Append predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Compute the confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(
            all_labels, all_preds, output_dict=True, zero_division=0
        )
        self.save_item(report, split)
        self.save_item(all_labels, split, 'labels')
        self.save_item(all_preds, split, 'preds')
        
        return cm, report, all_labels, all_preds

    def save_item(self, report, split, item='report'):
        path = os.path.join(self.out_dir, f"{split}_{item}.pkl")
        with open(path, "wb") as f:
            pkl.dump(report, f)
        
    def load_item(self, path, split, item='report'):
        path = os.path.join(self.out_dir, f"{split}_{item}.pkl")
        if os.path.exists(path):
            return pkl.load(open(path, "rb"))
        else:
            return None

class Args:
    # Provided values
    batch_size = 16
    epochs = 300
    model = "vit_large_patch16"
    input_size = 224
    drop_path = 0.2
    weight_decay = 0.05
    blr = 5e-4
    layer_decay = 0.65
    nb_classes = 5
    data_path = "/data/core-kind/fatemeh/data/IDRiD_data"
    task = "./out_files/retfound_mode_op/"
    finetune = "/data/core-kind/fatemeh/data/op_weights.pth"
    world_size = 1
    balance = 1
    loss = "cross_entropy"
    use_sigmoid = 0
    master_port = 56157
    nproc_per_node = 1

    # Default values from function
    accum_iter = 1
    clip_grad = None
    lr = None
    min_lr = 1e-6
    warmup_epochs = 10
    color_jitter = None
    aa = "rand-m1-mstd0.5-inc1"
    smoothing = 0.1
    reprob = 0.1
    remode = "pixel"
    recount = 1
    resplit = False
    mixup = 0
    cutmix = 0
    cutmix_minmax = None
    mixup_prob = 1.0
    mixup_switch_prob = 0.5
    mixup_mode = "batch"
    global_pool = True
    cls_token = False
    output_dir = "./output_dir"
    log_dir = "./output_dir"
    device = "cuda"
    seed = 0
    resume = ""
    start_epoch = 0
    eval = False
    dist_eval = False
    num_workers = 10
    pin_mem = True
    no_pin_mem = False
    local_rank = -1
    dist_on_itp = False
    dist_url = "env://"
    stats_source = "custom"
    transform = "custom"
