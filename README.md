# 🧠 Depression Detection from Body Pose using Multi-Scale Spatiotemporal Network (MSN)

This project implements a full pipeline to detect **depression** from **3D body pose sequences** captured using Kinect, using a **Multi-Scale Spatiotemporal Network (MSN)**.

---

## 🔗 GitHub Repository

👉 [https://github.com/kimiasedighi/depression-detection-MSN-model](https://github.com/kimiasedighi/depression-detection-MSN-model)

---

## 📁 Project Structure

```
.
├── prepare_data.py              # Preprocessing script
├── dataset.py                   # Custom PyTorch Dataset
├── msn_body.py                  # Multi-Scale Spatiotemporal Network
├── train.py                     # Training & evaluation
├── requirements.txt             # Python dependencies
├── processed_data/              # Output tensor files
├── 20250110_Participant_list_1.xlsx  # Metadata for labels
├── missing_*.txt                # Log files for missing data
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

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Format & Configuration

Your data should be organized like this:

```
📁 /path/to/json_root/
    └── <folder>/
        └── poses.json

📁 /path/to/raw_data/
    └── <subject_id>/
        ├── <folder>_app.csv
        └── <folder>_kinect_ts.txt
```

Edit the `CONFIG` dictionary in `prepare_data.py` to match your data paths:

```python
CONFIG = {
    "json_root_dir": "/your/path/to/3d-body-poses",
    "raw_data_dir": "/your/path/to/raw_data",
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

- Extracts and normalizes body poses
- Saves `.pt` files in `./processed_data`
- Logs missing files in:
  - `missing_label.txt`
  - `missing_poses_json.txt`
  - `missing_app_csv.txt`
  - `missing_kinect_ts.txt`

---

## 🏋️ Step 2: Train the MSN model

### Default:

```bash
python train.py
```

### With custom settings:

```bash
python train.py \
  --data_dir ./processed_data \
  --save_path best_msn_model.pth \
  --batch_size 16 \
  --lr 0.001 \
  --epochs 20
```

---

## 🧠 Model Architecture

**Multi-Scale Spatiotemporal Network (MSN)**

- **Input shape**: `[B, C=3, T=300, J=11]`
- **Multiple temporal branches** with different kernel sizes: `3, 5, 7`
- Each branch uses 1D convolution across time to capture features at different temporal scales
- Outputs are concatenated → fused → pooled → classified

```
             Input
               │
         ┌─────┴──────┐
         ▼            ▼
     Conv1D(k=3)   Conv1D(k=5)  ... (k=7)
         ▼            ▼
     Feature Map     Feature Map
         └─────┬──────┘
               ▼
          Concat + Fuse
               ▼
     Global Average Pooling
               ▼
           Fully Connected
               ▼
           Class logits
```

---

## 📊 Output Example

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

## 🚀 Full Pipeline in One Go

```bash
# Step 1: Preprocess
data
python prepare_data.py

# Step 2: Train & evaluate
python train.py --data_dir ./processed_data --epochs 20
```

---

## 📟 Requirements

All dependencies are in `requirements.txt`:

```
torch
pandas
numpy
scikit-learn
openpyxl
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 📬 Author

👩‍💻 **Kimia Sedighi**  
📌 GitHub: [@kimiasedighi](https://github.com/kimiasedighi)

---

## 📖 Citation

Adapted from the paper:

> A Deep Multiscale Spatiotemporal Network for Assessing Depression from Facial Dynamics by Wheidima Carneiro de Melo, Eric Granger and Abdenour Hadid 
> *IEEE Transactions on Affective Computing*

This version applies the method to **Kinect body pose data** instead of facial landmarks.

---

## 📜 License

MIT License.

---

## 💡 Contributing

PRs and ideas welcome! If you’d like to extend this model to other modalities (e.g., audio or facial features), feel free to fork and build on it.

---

**🧠 Let’s build better mental health detection tools with AI.**
