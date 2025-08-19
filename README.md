# Facial-recognition_ml
Face Recognition Fine-Tuning Project 

This project implements **face recognition with fine-tuning using Triplet Loss** on the [InceptionResnetV1](https://github.com/timesler/facenet-pytorch) architecture.  
The workflow includes:
- Dataset preparation (high/low resolution merge, person-wise split).
- Baseline evaluation on a pre-trained model.
- Fine-tuning with Triplet Loss.
- Post-finetune evaluation with ROC, PR, and similarity plots.

---

##  Project Folder Structure
face_recognition_project/
├── facedataset/ # Raw dataset (high + low resolution images)
│ ├── high_resolution/
│ └── low_resolution/
├── facedataset_split/ # Train/eval splits (80/20 per person)
│ ├── train/
│ └── eval/
├── outputs/
│ ├── checkpoints/ # Saved models (e.g., best_model.pth)
│ ├── plots/ # Baseline evaluation plots
│ └── post_eval_plots/ # Post-finetune evaluation plots
├── scripts/
│ ├── baseline_eval.py # Pre-finetune evaluation
│ ├── clean_images.py # Remove corrupted/unreadable images
│ ├── fine_tune.py # Fine-tuning with Triplet Loss
│ ├── post_eval.py # Post-finetune evaluation
│ ├── recreate_split.py # Dataset splitting (80/20 person-wise)
│ ├── test_triplet.py # Sanity check for Triplet dataset
│ ├── triplet_dataset.py # Custom dataset class (Anchor, Positive, Negative)
│ └── utils.py # Utility functions (loss plotting)
└── README.md

yaml
Copy
Edit

---

##  Setup Instructions

1. **Clone Repo & Install Requirements**
   ```bash
   git clone https://github.com/yourusername/face_recognition_project.git
   cd face_recognition_project/scripts
   pip install facenet-pytorch torchvision matplotlib seaborn scikit-learn
Prepare Dataset

bash
Copy
Edit
facedataset/
├── high_resolution/person1/
└── low_resolution/person1/
Create Train/Eval Split

bash
Copy
Edit
python3 recreate_split.py
Clean Dataset (remove corrupted images)

bash
Copy
Edit
python3 clean_images.py
 Usage (Scripts)
1. Sanity Check Dataset
bash
Copy
Edit
python3 test_triplet.py
2. Baseline Evaluation (Before Fine-tuning)
Computes ROC, PR curve, and cosine similarity histogram.

bash
Copy
Edit
python3 baseline_eval.py
Results saved in: outputs/plots/

3. Fine-Tune Model with Triplet Loss
bash
Copy
Edit
python3 fine_tune.py
Best model → outputs/checkpoints/best_model.pth

Training loss plot → outputs/plots/training_loss.png

4. Post-Finetune Evaluation
bash
Copy
Edit
python3 post_eval.py
Results saved in: outputs/post_eval_plots/

Sample Outputs
Type	Location
Best model	outputs/checkpoints/best_model.pth
Training Loss Plot	outputs/plots/training_loss.png
Evaluation Plots	outputs/plots/*.png
Post-Finetune Plots	outputs/post_eval_plots/*.png

🧩 Key Features
Person-wise dataset split (avoids identity leakage ✅).

Handles high/low resolution images seamlessly.

Triplet Loss based fine-tuning.

End-to-end evaluation (pre & post finetune).

Visualization of metrics (ROC, PR, similarity distribution, training loss).

Requirements
Python 3.8+

PyTorch

facenet-pytorch

torchvision

matplotlib, seaborn, scikit-learn

License
This project is released under the MIT License.
You are free to use, modify, and distribute this project with attribution.

Acknowledgments
FaceNet PyTorch implementation by David Sandberg.

Pre-trained InceptionResnetV1 on VGGFace2 dataset.

Scikit-learn for evaluation metrics.



