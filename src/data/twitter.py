from datasets import load_dataset
from torch.utils.data import DataLoader

dataset_name = "carblacac/twitter-sentiment-analysis"
batch_size = 2

dataset = load_dataset(dataset_name, split='train').with_format("torch")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

twitter_data = {"dataset": dataset, "dataloader": dataloader, "args":{'x': 'text', 'y': 'feeling'}}

if __name__ == "__main__":
    dt = dataset[0]


