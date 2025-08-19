from triplet_dataset import TripletFaceDataset
from torch.utils.data import DataLoader
from torchvision import transforms

dataset = TripletFaceDataset(
root_dir="../facedataset_split/train",
transform=transforms.ToTensor()
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

for i, (a, p, n) in enumerate(loader):
print(f"Batch {i}: Anchor {a.shape}, Positive {p.shape}, Negative {n.shape}")
if i == 1:
break
