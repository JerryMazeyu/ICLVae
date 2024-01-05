import numpy as np
import random
from abc import ABC, abstractmethod
from src.utils import create_dir, FileLogger, load_model
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

class Trainer(ABC):
    def __init__(self, model, data, conf) -> None:
        self.exp = input("exp name: ")
        self.prepare_path()
        self.prepare_model(model)
        self.prepare_conf(data, conf)
        self.prepare_other()
        self.prepare_para()

    def prepare_path(self):
        tmp, self.exp = create_dir(self.exp)
        print(f"Experiment's name is: {self.exp}")
        self.exp_path, self.ckpt_path, self.log_path, self.run_path = tmp['root'], tmp['ckpts'], tmp['logs'], tmp['runs']

    def prepare_model(self, model):
        self.llm, self.ori_llm, self.tokenizer, self.feature_map = model['vae_model'], model['ori_model'], model['tokenizer'], model['FEATURE_MAP']
        self.train_module = None
        try:
            tmp = self.llm
            for arg in model["args"]:
                tmp = getattr(tmp, arg)
            self.train_module = getattr(tmp, 'vae')
        except:
            self.train_module = self.llm.transformer.final_layernorm.vae
        try:
            self.use_score = model["use_score"]
        except:
            self.use_score = True
        try:
            self.score_list = model["SCORE_LIST"]
        except:
            self.score_list = None

    def prepare_conf(self, data, conf):
        self.logger = FileLogger(self.log_path)
        self.dataloader = data['dataloader']
        try:
            self.data_conf = data['args']
        except:
            self.data_conf = {'x': 'text', 'y': 'feeling'}
        self.x = self.data_conf['x']
        self.y = self.data_conf['y']
        self.device = conf['device']
        self.epochs = conf['epochs']
        self.lr = conf['lr']
        self.iteration = conf['iteration']
        self.iteration = 1e+5 if not self.iteration else self.iteration
        self.load_from = conf['load_from']
        self.lambda_ = conf['lambda_']
        self.lambda_scheduler = conf['lambdascheduler']

    def prepare_other(self):
        self.writer_ = SummaryWriter(self.log_path)
        self.optimizer = torch.optim.AdamW(self.train_module.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-4)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)

    @abstractmethod
    def prepare_para(self):
        pass

    @abstractmethod
    def preprocess_data(self, batch, *args, **kwargs):
        pass

    def set_seed(self, seed_value):
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed_value)
        random.seed(seed_value)

    @abstractmethod
    def run(self):
        pass


class Visualizer(ABC):
    def __init__(self, model, data, conf) -> None:
        self.exp = input("exp name: ")
        self.prepare_path()
        self.prepare_model(model)
        self.prepare_conf(data, conf)

    def prepare_path(self):
        tmp, self.exp = create_dir(self.exp)
        print(f"Experiment's name is: {self.exp}")
        self.exp_path, _, self.log_path, self.run_path = tmp['root'], tmp['ckpts'], tmp['logs'], tmp['runs']

    def prepare_model(self, model):
        self.llm, self.ori_llm, self.tokenizer, self.feature_map = model['vae_model'], model['ori_model'], model['tokenizer'], model['FEATURE_MAP']

    def prepare_conf(self, data, conf):
        self.logger = FileLogger(self.log_path)
        self.dataloader = data['dataloader']
        try:
            self.data_conf = data['args']
        except:
            self.data_conf = {'x': 'text', 'y': 'feeling'}
        self.x = self.data_conf['x']
        self.y = self.data_conf['y']
        self.device = conf['device']
        self.iteration = conf['iteration']
        self.iteration = 1e+5 if not self.iteration else self.iteration
        self.load_from = conf['load_from']

    @abstractmethod
    def preprocess_data(self, batch, *args, **kwargs):
        pass

    def set_seed(self, seed_value):
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed_value)
        random.seed(seed_value)

    @abstractmethod
    def run(self):
        pass
