import os

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

_DATASETS = {"CIFAR10": CIFAR10, "CIFAR100": CIFAR100}


# Function to load a dataset and preprocess it
def load_dataset(
	dataset_name: str = "CIFAR10",
	batch_size: int = 32,
	shuffle: bool = True,
	root: str = "./data",
	subset_size: int = None,
):
	if dataset_name not in _DATASETS:
		raise ValueError(f"Dataset '{dataset_name}' is not available yet! Choose from {list(_DATASETS)}.")

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])

	os.makedirs(root, exist_ok=True)
	dataset_cls = _DATASETS[dataset_name]
	dataset = dataset_cls(root=root, train=True, download=True, transform=transform)

	if subset_size is not None:
		dataset = Subset(dataset, range(min(subset_size, len(dataset))))

	data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

	print(f"Dataset: {dataset_name} is loaded| len: {len(data_loader)}")
	return data_loader
