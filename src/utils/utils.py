import os
import datetime
import uuid
import torch
import warnings
import re

def create_dir(name='', root="/home/ubuntu/LLMs/FineTuneLLMs/mzy/exp"):
    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_name = uuid.uuid4().hex
    directory_name = name if name != '' else f"{date_time}_{random_name[0:5]}"
    path_ = os.path.join(root, directory_name)
    os.makedirs(path_, exist_ok=True)
    paths = {'root': path_, 'logs': os.path.join(path_, 'logs'), 'runs': os.path.join(path_, 'runs'), 'ckpts': os.path.join(path_, 'ckpts')}
    os.makedirs(paths['logs'], exist_ok=True)
    os.makedirs(paths['runs'], exist_ok=True)
    os.makedirs(paths['ckpts'], exist_ok=True)
    return paths, directory_name

class FileLogger():
    def __init__(self, path_):
        self.path = os.path.join(path_, 'log.txt')
        # os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.file = open(self.path, 'a')

    def log(self, message):
        self.file.write(message + '\n')
        print(message)

    def __del__(self):
        self.file.close()

def load_model(exp, cls):
    if os.path.isfile(exp):
        latest_checkpoint = exp
        print(f"Pretrained checkpoint loaded: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        try:
            cls.llm.load_state_dict(checkpoint['model_state_dict'])
        except:
            warnings.warn("Load model state dict failed!", RuntimeWarning)
        try:
            cls.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            warnings.warn("Load optimizer state dict failed!", RuntimeWarning)
        cls.step = checkpoint['step']
    else:
        ckpt_dir = os.path.join(exp, 'ckpts')
        checkpoint_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
        sorted_checkpoints = sorted(checkpoint_files, key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)), reverse=True)
        if sorted_checkpoints:
            latest_checkpoint = os.path.join(ckpt_dir, sorted_checkpoints[0])
            print(f"Pretrained checkpoint loaded: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint)
            try:
                cls.llm.load_state_dict(checkpoint['model_state_dict'])
            except:
                warnings.warn("Load model state dict failed!", RuntimeWarning)
            try:
                cls.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                warnings.warn("Load optimizer state dict failed!", RuntimeWarning)
            cls.step = checkpoint['step']
        else:
            raise ValueError(f"There is no checkpoints found in {exp}.")



def clean_and_link_checkpoints(directory, keep=3, link_name="best.pth"):
    if not os.path.isdir(directory):
        print(f"Dir '{directory}' not exist!")
        return
    checkpoint_pattern = re.compile(r'epoch-(\d+)_step-(\d+)\.pth')
    def sort_key(filename):
        match = checkpoint_pattern.match(filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return 0, 0

    checkpoints = [f for f in os.listdir(directory) if checkpoint_pattern.match(f)]
    checkpoints.sort(key=sort_key, reverse=True)
    for checkpoint in checkpoints[keep:]:
        os.remove(os.path.join(directory, checkpoint))
        print(f"Remove file: {checkpoint}")
    if checkpoints:
        latest_checkpoint = os.path.join(directory, checkpoints[0])
        link_path = os.path.join(directory, link_name)
        if os.path.islink(link_path):
            os.remove(link_path)
        os.symlink(latest_checkpoint, link_path)
        print(f"Softlink '{link_name}' to '{latest_checkpoint}'")



if __name__ == '__main__':
    clean_and_link_checkpoints("/home/ubuntu/LLMs/FineTuneLLMs/mzy/exp/cola_train_align_20240102/ckpts")