import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load fine-tuned model
model = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(device)
model.load_state_dict(torch.load('../outputs/checkpoints/best_model.pth'))

# Paths
val_dir = '../facedataset_split/eval' # use only eval set
output_dir = '../outputs/post_eval_plots'
os.makedirs(output_dir, exist_ok=True)

# Image preprocessing
transform = transforms.Compose([
transforms.Resize((160, 160)),
transforms.ToTensor(),
])

def load_image(path):
try:
img = Image.open(path).convert('RGB')
return transform(img)
except Exception as e:
print(f"[Warning] Skipping image: {path} — {e}")
return None

def get_image_pairs(data_dir, num_pairs=100):
same_pairs = []
diff_pairs = []
classes = os.listdir(data_dir)

for _ in range(num_pairs):
cls = random.choice(classes)
imgs = [f for f in os.listdir(os.path.join(data_dir, cls)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if len(imgs) >= 2:
pair = random.sample(imgs, 2)
same_pairs.append((os.path.join(data_dir, cls, pair[0]),
os.path.join(data_dir, cls, pair[1]), 1))

cls1, cls2 = random.sample(classes, 2)
img1 = random.choice(os.listdir(os.path.join(data_dir, cls1)))
img2 = random.choice(os.listdir(os.path.join(data_dir, cls2)))
diff_pairs.append((os.path.join(data_dir, cls1, img1),
os.path.join(data_dir, cls2, img2), 0))
return same_pairs + diff_pairs

# Extract embeddings
pairs = get_image_pairs(val_dir, num_pairs=100)
embeddings = []

for path1, path2, label in tqdm(pairs, desc="Evaluating pairs"):
img1 = load_image(path1)
img2 = load_image(path2)
if img1 is None or img2 is None:
continue

img1 = img1.unsqueeze(0).to(device)
img2 = img2.unsqueeze(0).to(device)

emb1 = F.normalize(model(img1))
emb2 = F.normalize(model(img2))
sim = F.cosine_similarity(emb1, emb2).item()
embeddings.append((sim, label))

# Plot metrics
sims, labels = zip(*embeddings)
fpr, tpr, _ = roc_curve(labels, sims)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(labels, sims)

# Accuracy
threshold = 0.5
preds = [1 if s >= threshold else 0 for s in sims]
acc = accuracy_score(labels, preds)

# ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()
plt.savefig(os.path.join(output_dir, 'roc_curve_post.png')) # ✅ UPDATED

# PR Curve
plt.figure()
plt.plot(recall, precision, label='PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig(os.path.join(output_dir, 'pr_curve_post.png')) # ✅ UPDATED

# Similarity Histogram
same = [sim for sim, l in embeddings if l == 1]
diff = [sim for sim, l in embeddings if l == 0]

plt.figure()
sns.histplot(same, color='green', label='Same', kde=True)
sns.histplot(diff, color='red', label='Different', kde=True)
plt.title('Similarity Distribution')
plt.xlabel('Cosine Similarity')
plt.legend()
plt.savefig(os.path.join(output_dir, 'similarity_hist_post.png')) # ✅ UPDATED

print(f"\n✅ Post-Finetune Accuracy @ threshold 0.5: {acc:.2%}")
