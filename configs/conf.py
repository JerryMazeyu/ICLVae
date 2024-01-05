import torch
from src.scheduler import *

baseconf = {
    "device": torch.device("cuda:0"),
    "epochs": 2,
    "lr": 1e-5,
    "iteration": None,
    "lambda_": 0.8,
    "lambdascheduler": scheduler_8_linear_1,
    "load_from": "/home/ubuntu/LLMs/FineTuneLLMs/mzy/exp/train-from-scratch-again/ckpts/best.pth",
}

visiualconf = {
    "device": torch.device("cuda"),
    "iteration": 50,  # self.step数量上限
    "load_from": "/home/ubuntu/LLMs/FineTuneLLMs/model/2023-12-27_23-53-01/epoch-1_step-3500.pth"
}

# finetuneconf = baseconf = {
#     "device": torch.device("cuda:0"),
#     "epochs": 1,
#     "lr": 1e-5,
#     "iteration": None,
#     "lambda_": 0.8,
#     "lambdascheduler": scheduler_8_linear_1,
#     "load_from": "/home/ubuntu/LLMs/FineTuneLLMs/mzy/exp/train-from-scratch-again/ckpts/best.pth",
# }

# baseconf = {
#     "model_name": "THUDM/chatglm-6b",
#     "use_auth_token": "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",
#     "dataset_name": "carblacac/twitter-sentiment-analysis",
#     "split": "test",
#     "batch_size": 8,
#     "model_y_config": {
#         "input_dim" : 4096,
#         "hidden_dim" : 512,
#         "latent_dim" : 4096,
#         "n_layers" : 8,
#         "heads" : 16
#     },
#     "model_y_pretrained": None,
#     "checkpoints_path": "/home/ubuntu/LLMs/FineTuneLLMs/model/model_align/",
#     "lr": 1e-5,
#     "epochs": 5,
#     "iteration": 10,
#     "lambda_": 0.8,
#     "logfile": "logs/log_align3",
# }