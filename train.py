from src import PatientEventsDataset, SameFileBatchSampler, DBTransformer
from tqdm import trange
import deeppy as dp
from deeppy import LearnFrame
from src.loader import DatasetLoader
import torch
import time

import os


os.makedirs("/home/baris/torch-tmp", exist_ok=True)
os.environ["TMPDIR"] = "/home/baris/torch-tmp"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/home/baris/torch-tmp/inductor"
os.environ["TRITON_CACHE_DIR"] = "/home/baris/torch-tmp/triton"


dp.env_config.use_amp = True
dp.env_config.torch_compile = True
dp.env_config.torch_compile_args = {"dynamic" : True}
dp.env_config.xai_optimizer_log_freq = 200
print(dp.env_config.device)
print(dp.env_config.log_dir)
print(dp.env_config.checkpoint_dir)
torch.set_float32_matmul_precision('high')

PATH =  "/home/baris/database_transformer/dataset/ml"
dictionary_path = "/home/baris/database_transformer/dataset/ml/vocab.pt"

batch_size = 32
lr = 5e-5
lr = 1e-4
#epochs = 1.6 * 2
epochs = 2
steps = int((4_000_000 / batch_size) * epochs)

context_size = 7000

Scheduler_params = {
                "scheduler" : torch.optim.lr_scheduler.OneCycleLR,
                "auto_step":True,
                 "max_lr": lr,
                "total_steps": int(steps),
                "pct_start": 0.2,
                "anneal_strategy": "cos",
                "cycle_momentum": True,
                "base_momentum": 0.85,
                "max_momentum": 0.95,
                "div_factor": 25,
                "final_div_factor": 10000,
                "three_phase": False,
                "last_epoch": -1,
}

optimizer_params = {
    "optimizer" : torch.optim.AdamW,
    "optimizer_args" : {
        "lr" : lr,
        "weight_decay" : 1e-2,
        "fused" : True
    },
    "clipper" : torch.nn.utils.clip_grad_norm_,
    "clipper_params" : {"max_norm" : 5.0},
    "scheduler_params" : Scheduler_params,
    "gradient_accumulation_steps" : 1,
}

lazy_index= False
drop_last = True

args = {
    "path": PATH,
    "context_size": context_size,
    "threads": 32,
    "mask_rate" : 0.15
}

dataset_train = PatientEventsDataset(**args, lazy_index=lazy_index)
dataset_train.init_mode("train")

dataset_valid = PatientEventsDataset(**args, lazy_index=True)
dataset_valid.init_mode("valid")

dataset_test = PatientEventsDataset(**args, lazy_index=lazy_index)
dataset_test.init_mode("test")

bs_train = SameFileBatchSampler(dataset=dataset_train, batch_size=batch_size, shuffle=True, drop_last=drop_last)
bs_valid = SameFileBatchSampler(dataset=dataset_valid, batch_size=batch_size, shuffle=True, drop_last=drop_last)
bs_test = SameFileBatchSampler(dataset=dataset_test, batch_size=batch_size, shuffle=True, drop_last=drop_last)


dataloader_args = {
    "num_workers": 0,
    "pin_memory": True,
}

dataloader = DatasetLoader(dataset_train, dataset_test, dataset_valid, bs_train,  bs_test, bs_valid, batch_size = 0, dataloader_args = dataloader_args)

nhead = 8
d_model = int(nhead*64)
num_encoder_layers = 4
dropout = 0.1
activation = "gelu"
num_gaussians = 10

model = DBTransformer(PATH + "/vocab.pt", optimizer_params,
	 			num_frequencies=10, num_gaussians=num_gaussians,
				d_model = d_model, nhead= nhead, num_encoder_layers = num_encoder_layers, dim_feedforward = 4 * d_model, 
				context_size=context_size, dropout = dropout, activation = activation, 
                last_layer_norm = True, use_checkpointing=True)

print("Total trainable parameters:", sum(p.numel() for net in model.nets for p in net.parameters() if p.requires_grad))


lf = LearnFrame(model,dataloader)
lf.train(100, steps//5, steps,1)

print("Training completed. Saving model...")