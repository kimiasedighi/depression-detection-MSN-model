# 🧠 Depression Detection from Body Pose using Multi-Scale Spatiotemporal Network (MSN)

This project implements a full pipeline to detect **depression** from **3D body pose sequences** captured using Kinect, using a **Multi-Scale Spatiotemporal Network (MSN)** in PyTorch.

The system includes:
- Data preprocessing and extraction from raw pose sequences and labels
- A custom PyTorch dataset class
- A Multi-Scale CNN-based architecture for spatiotemporal body dynamics
- Training and evaluation pipeline with classification metrics

---

## 🔗 GitHub Repository

👉 [https://github.com/kimiasedighi/depression-detection-MSN-model](https://github.com/kimiasedighi/depression-detection-MSN-model)

---

## 📁 Project Structure

```
.
├── prepare_data.py              # Preprocesses raw Kinect pose and label data
├── dataset.py                   # Custom PyTorch Dataset loader for .pt files
├── msn_body.py                  # Multi-Scale Spatiotemporal Network architecture
├── train.py                     # Model training and evaluation script
├── requirements.txt             # List of dependencies
├── processed_data/              # Output .pt tensors (C,T,J format)
├── 20250110_Participant_list_1.xlsx  # Excel sheet with label metadata
├── missing_*.txt                # Logs for missing participants/data types
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/kimiasedighi/depression-detection-MSN-model.git
cd depression-detection-MSN-model
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv msn_env
source msn_env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Format & Configuration

Your dataset must include:

```
📁 /json_root_dir/
    └── <folder>/
        └── poses.json              # Contains body joint positions (3D)

📁 /raw_data_dir/
    └── <subject_id>/
        ├── <folder>_app.csv        # Event and timestamp logs
        └── <folder>_kinect_ts.txt  # Kinect frame timestamps
```

Edit paths in `prepare_data.py`:
```python
CONFIG = {
    "json_root_dir": "/path/to/json_root",
    "raw_data_dir": "/path/to/raw_data",
    "label_file": "20250110_Participant_list_1.xlsx",
    "output_dir": "./processed_data",
    ...
}
```

---

## 🧹 Step 1: Preprocess the data

```bash
python prepare_data.py
```

This script will:
- Load body pose data from `poses.json`
- Extract timestamped events from `app.csv`
- Match with Kinect timestamps in `.txt`
- Normalize joint positions
- Save body pose tensors in shape **(C=3, T=300, J=11)**
- Log any missing files per participant in `missing_*.txt`

### Output:
```
processed_data/
├── 823_t2_20240701.pt
├── 824_t2_20240702.pt
...
```
Each `.pt` file contains a dictionary with:
```python
{
  "data": torch.FloatTensor (C, T, J),
  "label": 0 or 1  # healthy or depressed
}
```

---

## 🧠 Step 2: Model Architecture — MSN (Multi-Scale Network)

Defined in `msn_body.py`, the model uses **multiple temporal convolutional branches** to extract features at different scales.

### Architecture Overview:
- Input: `[B, 3, 300, 11]` (B=batch, C=channels, T=frames, J=joints)
- 3 parallel Conv1D branches with kernel sizes `[3,5,7]`
- Feature fusion (Concat + Conv1x1)
- Global Average Pooling
- Fully connected softmax layer

### Why multi-scale?
- Depression affects **body dynamics** across different time spans
- Multi-kernel convolutions capture short, medium, and long temporal patterns

---

## 🏋️ Step 3: Train the Model

### Default training:
```bash
python train.py
```

### Customize training:
```bash
python train.py \
  --data_dir ./processed_data \
  --save_path best_msn_model.pth \
  --batch_size 16 \
  --lr 0.001 \
  --epochs 20
```

### Output:
- Best model saved as `best_msn_model.pth`
- Classification report and confusion matrix

### Example:
```bash
Epoch 12/20 - Loss: 0.1532 - Val Acc: 94.12%
✅ Saved best model!

🎯 Final Test Accuracy: 87.50%

Classification Report:
               precision    recall  f1-score   support
    Healthy        0.86       0.88       0.87        16
  Depressed        0.89       0.86       0.87        16

Confusion Matrix:
[[14  2]
 [ 2 14]]
```

---

## 💡 Code Breakdown

### `prepare_data.py`
- Loads labels from Excel (`Sheet3`)
- Matches IDs based on `CRADK` or `ADK` condition
- Extracts relevant time segments based on labels: `ei_01` to `ei_10`, or labels containing both `training` and `feedback`
- Matches time ranges to Kinect timestamps
- Extracts pose sequences
- Pads/truncates to 300 frames

### `dataset.py`
```python
class PoseDataset(Dataset):
    def __getitem__(self, idx):
        item = torch.load(self.data_files[idx])
        return item["data"], item["label"]
```

### `msn_body.py`
- Defines `MultiScaleBlock` with Conv1D over time
- Combines parallel convolutions into a fused representation
- Final prediction through `Linear`

### `train.py`
- Loads dataset from `.pt` files
- Splits into train/val/test
- Trains model with evaluation per epoch
- Prints classification report at the end

---

## 🚀 Run Everything in One Go

```bash
# Step 1: Preprocess
python prepare_data.py

# Step 2: Train & evaluate
python train.py --data_dir ./processed_data --epochs 20
```

---

## 📦 Requirements

Install via:
```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
torch
pandas
numpy
scikit-learn
openpyxl
```

---

## 📬 Author

👩‍💻 **Kimia Sedighi**  
🔗 GitHub: [@kimiasedighi](https://github.com/kimiasedighi)

---

## 📖 Citation

Adapted from the paper:
> A Deep Multiscale Spatiotemporal Network for Assessing Depression From Facial Dynamics  
> *IEEE Transactions on Affective Computing*

This repo applies the method to **Kinect body pose data**.

---

## 📜 License
MIT License (or add your own)

---

## 💡 Contributing

Pull requests and ideas welcome! Want to extend this to facial or audio data? Feel free to fork and build on it.

---

**🧠 Let’s build better mental health detection tools with AI.**
