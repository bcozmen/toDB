from src import PatientEventsDataset, SameFileBatchSampler, DBTransformer
import deeppy as dp
import torch

dp.env_config.use_amp = True
dp.env_config.torch_compile = True
dp.env_config.torch_compile_args = {"dynamic" : True}
dp.env_config.xai_optimizer_log_freq = 200
dp.env_config.device = torch.device("cpu")
print(dp.env_config.device)
print(dp.env_config.log_dir)
print(dp.env_config.checkpoint_dir)
torch.set_float32_matmul_precision('high')

nhead = 8
d_model = int(nhead*64)
num_encoder_layers = 12
dropout = 0.1
activation = "gelu"
num_gaussians = 10
context_size = 2048
lr = 1e-4
steps = 100000

PATH =  "/home/baris/database_transformer/dataset/ml"

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

model = DBTransformer(PATH + "/vocab.pt", optimizer_params,
	 			num_frequencies=10, num_gaussians=num_gaussians,
				d_model = d_model, nhead= nhead, num_encoder_layers = num_encoder_layers, dim_feedforward = 4 * d_model, 
				context_size=context_size, dropout = dropout, activation = activation, 
                last_layer_norm = True, use_checkpointing=True)
#model.load("/home/baris/database_transformer/exp2/checkpoints/160000")
model.eval()





