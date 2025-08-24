# prepare_data.py
import os
import pandas as pd
import json
import torch
import numpy as np
from datetime import datetime

# --- Configurable Parameters ---
CONFIG = {
    "json_root_dir": "/home/janus/iwso-datasets/t2-3d-body-poses",
    "raw_data_dir": "/home/vault/empkins/tpD/D02/RCT/raw_data",
    "label_file": "20250110_Participant_list_1.xlsx",
    "output_dir": "./processed_data",
    "sheet_name": "Sheet3",
    "label_column_depressed": "Bedingung",
    "label_column_healthy": "Bedingung.1",
    "id_column_depressed": "ID",
    "id_column_healthy": "ID.1",
    "label_keywords": ["CRADK", "ADK"],
    "num_joints": 11,
    "channels": 3,
    "frame_len": 300
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# --- Load Label Dictionary ---
sheet3_df = pd.read_excel(CONFIG["label_file"], sheet_name=CONFIG["sheet_name"], header=2, engine="openpyxl")
sheet3_df.columns = (
    sheet3_df.columns.astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

assert CONFIG["label_column_depressed"] in sheet3_df.columns, f"❌ Column '{CONFIG['label_column_depressed']}' not found"

# Filter groups
pattern = "|".join(CONFIG["label_keywords"])
depressed_ids = sheet3_df[
    sheet3_df[CONFIG["label_column_depressed"]].str.contains(pattern, case=False, na=False)
][CONFIG["id_column_depressed"]].dropna().astype(str).str.split(".").str[0]

healthy_ids = sheet3_df[
    sheet3_df[CONFIG["label_column_healthy"]].str.contains(pattern, case=False, na=False)
][CONFIG["id_column_healthy"]].dropna().astype(str).str.split(".").str[0]

labels_dict = {
    **{id_: 1 for id_ in depressed_ids},
    **{id_: 0 for id_ in healthy_ids}
}
labels_dict.pop("ID", None)

# --- Preprocessing Parameters ---
NUM_JOINTS = CONFIG["num_joints"]
CHANNELS = CONFIG["channels"]
FRAME_LEN = CONFIG["frame_len"]

def normalize_joints(joints):
    return joints - joints[0]

def log_status_file(log_file, folder_name):
    with open(log_file, 'a') as f:
        f.write(f"{folder_name}\n")

def process_participant(folder_name):
    print(f"\nProcessing {folder_name}...")
    subject_id = folder_name.split("_")[0]

    log_files = {
        "missing_label": "missing_label.txt",
        "missing_poses": "missing_poses_json.txt",
        "missing_app": "missing_app_csv.txt",
        "missing_kinect": "missing_kinect_ts.txt",
        "ok": "everything_ok.txt"
    }

    poses_json = os.path.join(CONFIG["json_root_dir"], folder_name, "poses.json")
    app_csv = os.path.join(CONFIG["raw_data_dir"], subject_id, f"{folder_name}_app.csv")
    kinect_ts = os.path.join(CONFIG["raw_data_dir"], subject_id, f"{folder_name}_kinect_ts.txt")

    if subject_id not in labels_dict:
        print(f"  ⚠️ Skipping {folder_name} — no label found.")
        log_status_file(log_files["missing_label"], folder_name)
        return

    if not os.path.exists(poses_json):
        print(f"  ❌ Skipping {folder_name} — missing poses_json.")
        log_status_file(log_files["missing_poses"], folder_name)
        return

    if not os.path.exists(app_csv):
        folder_dir = os.path.join(CONFIG["raw_data_dir"], subject_id)
        csv_files = [f for f in os.listdir(folder_dir) if "app" in f and f.endswith(".csv")]
        if not csv_files:
            log_status_file(log_files["missing_app"], folder_name)
            return
        try:
            dfs = [pd.read_csv(os.path.join(folder_dir, f)) for f in csv_files]
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df['label'] = merged_df['label'].astype(str).str.strip('"')
            merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'].astype(str).str.strip('"'), errors='coerce')
            merged_df = merged_df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            app_csv = os.path.join(CONFIG["output_dir"], f"{folder_name}_merged_app.csv")
            merged_df.to_csv(app_csv, index=False)
        except Exception as e:
            print(f"  ❌ Failed to merge app CSVs: {e}")
            return

    if not os.path.exists(kinect_ts):
        folder_dir = os.path.join(CONFIG["raw_data_dir"], subject_id)
        txt_files = [f for f in os.listdir(folder_dir) if f.lower().endswith(".txt") and ("kinect" in f.lower())]
        if not txt_files:
            log_status_file(log_files["missing_kinect"], folder_name)
            return
        kinect_ts = os.path.join(folder_dir, txt_files[0])

    log_status_file(log_files["ok"], folder_name)
    label = labels_dict[subject_id]

    try:
        df = pd.read_csv(app_csv)
        df['label'] = df['label'].astype(str).str.strip('"')
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str).str.strip('"'), errors='coerce')
        df = df.dropna(subset=['timestamp'])

        filtered_rows = df[
            df['label'].isin([f"ei_{str(i).zfill(2)}" for i in range(1, 11)]) |
            (df['label'].str.contains("training", case=False) & df['label'].str.contains("feedback", case=False))
        ].sort_values('timestamp').reset_index()

        if filtered_rows.empty:
            print(f"  ⚠️ No matching labels found in app CSV.")
            return

        with open(kinect_ts, "r") as f:
            lines = [line.strip() for line in f if "Start time" not in line]
        frame_times = [datetime.strptime(line, "%Y-%m-%d %H:%M:%S.%f") for line in lines]

        with open(poses_json, "r") as f:
            frames = json.load(f).get("frames", [])

        selected_frames = []
        for _, row in filtered_rows.iterrows():
            start_time = row['timestamp']
            full_index = df[df['timestamp'] == start_time].index
            if full_index.empty or full_index[0] + 1 >= len(df):
                continue
            end_time = df.iloc[full_index[0] + 1]['timestamp']
            try:
                start_frame = next(i for i, t in enumerate(frame_times) if t >= start_time)
                end_frame = next(i for i, t in reversed(list(enumerate(frame_times))) if t <= end_time)
                selected_frames.extend(frames[start_frame:end_frame + 1])
            except StopIteration:
                continue

        sequence = []
        for frame in selected_frames:
            joints = frame.get("poses", [])
            coords = np.zeros((NUM_JOINTS, CHANNELS))
            for joint in joints:
                jid = joint.get("joint")
                if jid is not None and jid < NUM_JOINTS:
                    coords[jid] = [joint["x_3d"], joint["y_3d"], joint["z_3d"]]
            sequence.append(normalize_joints(coords))

        sequence = np.stack(sequence) if sequence else np.zeros((1, NUM_JOINTS, CHANNELS))
        if sequence.shape[0] < FRAME_LEN:
            pad = np.zeros((FRAME_LEN - sequence.shape[0], NUM_JOINTS, CHANNELS))
            sequence = np.concatenate((sequence, pad), axis=0)
        else:
            sequence = sequence[:FRAME_LEN]

        tensor = torch.tensor(sequence.transpose(2, 0, 1), dtype=torch.float32)
        torch.save({"data": tensor, "label": label}, os.path.join(CONFIG["output_dir"], f"{folder_name}.pt"))
        print(f"  ✅ Saved tensor for {folder_name}")
    except Exception as e:
        print(f"  ❌ Error processing {folder_name}: {e}")

# --- Run for All ---
for folder in sorted(os.listdir(CONFIG["json_root_dir"])):
    if os.path.isdir(os.path.join(CONFIG["json_root_dir"], folder)):
        process_participant(folder)
