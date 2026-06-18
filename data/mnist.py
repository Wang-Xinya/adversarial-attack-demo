from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def ensure_mnist_downloaded(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    tfm = transforms.ToTensor()
    for train in (True, False):
        datasets.MNIST(str(data_dir), train=train, download=True, transform=tfm)
    print(f"[data] MNIST ready: {data_dir.resolve()}")

def get_mnist_loader(data_dir: Path, batch_size: int, train: bool, num_workers: int = 0):
    dataset = datasets.MNIST(str(data_dir), train=train, download=True,
                             transform=transforms.ToTensor())
    shuffle = train
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)