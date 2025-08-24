# prepare_data.py
import os
import pandas as pd
import json
import torch
import numpy as np
from datetime import datetime

# --- Config Paths ---
json_root_dir = "/home/janus/iwso-datasets/t2-3d-body-poses"
raw_data_dir = "/home/vault/empkins/tpD/D02/RCT/raw_data"
label_file = "20250110_Participant_list_1.xlsx"
output_dir = "./processed_data"
os.makedirs(output_dir, exist_ok=True)

# --- Load Label Dictionary from Excel ---
sheet3_df = pd.read_excel(label_file, sheet_name="Sheet3", header=2, engine="openpyxl")

sheet3_df.columns = (
    sheet3_df.columns.astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

print("✅ Cleaned column names:", sheet3_df.columns.tolist())
assert "Bedingung" in sheet3_df.columns, "❌ Column 'Bedingung' not found"

# Filter depressed group where Bedingung contains 'CRADK' or 'ADK'
depressed_filtered = sheet3_df[
    sheet3_df["Bedingung"].str.contains("CRADK|ADK", case=False, na=False)
]
depressed_ids = depressed_filtered["ID"].dropna().astype(str).str.split('.').str[0]

# Filter healthy group where Bedingung contains 'CRADK' or 'ADK'
healthy_filtered = sheet3_df[
    sheet3_df["Bedingung.1"].str.contains("CRADK|ADK", case=False, na=False)
]
healthy_ids = healthy_filtered["ID.1"].dropna().astype(str).str.split('.').str[0]

# Build label dictionary (1 for depressed, 0 for healthy)
labels_dict = {
    **{id_: 1 for id_ in depressed_ids},
    **{id_: 0 for id_ in healthy_ids}
}
labels_dict.pop("ID", None)

# --- Preprocessing Parameters ---
NUM_JOINTS = 11
CHANNELS = 3
FRAME_LEN = 300

def normalize_joints(joints):
    root = joints[0]
    return joints - root

def log_status_file(log_file, folder_name):
    with open(log_file, 'a') as f:
        f.write(f"{folder_name}\n")

def process_participant(folder_name):
    print(f"\nProcessing {folder_name}...")

    subject_id = folder_name.split("_")[0]

    missing_label_log = "missing_label.txt"
    missing_poses_json_log = "missing_poses_json.txt"
    missing_app_csv_log = "missing_app_csv.txt"
    missing_kinect_ts_log = "missing_kinect_ts.txt"
    everything_ok_log = "everything_ok.txt"

    poses_json = os.path.join(json_root_dir, folder_name, "poses.json")
    app_csv = os.path.join(raw_data_dir, subject_id, f"{folder_name}_app.csv")
    kinect_ts = os.path.join(raw_data_dir, subject_id, f"{folder_name}_kinect_ts.txt")

    if subject_id not in labels_dict:
        print(f"  ⚠️ Skipping {folder_name} — no label found.")
        log_status_file(missing_label_log, folder_name)
        return

    if not (os.path.exists(poses_json)):
        print(f"  ❌ Skipping {folder_name} — missing poses_json.")
        log_status_file(missing_poses_json_log, folder_name)
        return

    if not os.path.exists(app_csv):
        print(f"  ⚠️ {folder_name} — app_csv not found, attempting to merge available app CSVs.")
        folder_dir = os.path.join(raw_data_dir, subject_id)
        csv_files = [f for f in os.listdir(folder_dir) if "app" in f and f.endswith(".csv")]
        if not csv_files:
            print(f"  ❌ No alternative app CSVs found in {folder_dir}")
            log_status_file(missing_app_csv_log, folder_name)
            return

        try:
            dfs = []
            for csv_file in csv_files:
                path = os.path.join(folder_dir, csv_file)
                df_temp = pd.read_csv(path)
                dfs.append(df_temp)

            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df['label'] = merged_df['label'].astype(str).str.strip('"')
            merged_df['timestamp'] = merged_df['timestamp'].astype(str).str.strip('"')
            merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'], errors='coerce')
            merged_df = merged_df.dropna(subset=['timestamp'])
            merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
            merged_app_path = os.path.join(output_dir, f"{folder_name}_merged_app.csv")
            merged_df.to_csv(merged_app_path, index=False)
            app_csv = merged_app_path
            print(f"  ✅ Merged {len(csv_files)} app CSVs — using merged version.")
        except Exception as e:
            print(f"  ❌ Failed to merge alternative app CSVs: {e}")
            return

    if not os.path.exists(kinect_ts):
        print(f"  ⚠️ {folder_name} — kinect_ts not found, searching for alternative .txt files.")
        folder_dir = os.path.join(raw_data_dir, subject_id)
        try:
            txt_files = [
                f for f in os.listdir(folder_dir)
                if f.lower().endswith(".txt") and ("kinect" in f.lower() or "kinnect" in f.lower())
            ]
            if not txt_files:
                print(f"  ❌ No alternative Kinect .txt files found in {folder_dir}")
                log_status_file(missing_kinect_ts_log, folder_name)
                return
            kinect_ts = os.path.join(folder_dir, txt_files[0])
            print(f"  ✅ Using alternative Kinect file: {txt_files[0]}")
        except Exception as e:
            print(f"  ❌ Error while searching for Kinect .txt files: {e}")
            return

    log_status_file(everything_ok_log, folder_name)
    label = labels_dict[subject_id]

    try:
        df = pd.read_csv(app_csv)
        df['label'] = df['label'].astype(str).str.strip('"')
        df['timestamp'] = df['timestamp'].astype(str).str.strip('"')
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])

        filtered_rows = df[
            df['label'].isin([f"ei_{str(i).zfill(2)}" for i in range(1, 11)]) |
            (
                df['label'].str.contains("training", case=False, na=False) &
                df['label'].str.contains("feedback", case=False, na=False)
            )
        ]

        if filtered_rows.empty:
            print(f"  ⚠️ No matching labels found in app CSV.")
            return

        df = df.sort_values('timestamp').reset_index(drop=True)
        filtered_rows = filtered_rows.sort_values('timestamp').reset_index()

        selected_frames = []
        with open(kinect_ts, "r") as f:
            lines = [line.strip() for line in f if "Start time" not in line]
        frame_times = [datetime.strptime(line, "%Y-%m-%d %H:%M:%S.%f") for line in lines]

        with open(poses_json, "r") as f:
            json_data = json.load(f)
        frames = json_data.get("frames", [])

        for _, row in filtered_rows.iterrows():
            start_time = row['timestamp']
            full_index = df[df['timestamp'] == start_time].index
            if full_index.empty or full_index[0] + 1 >= len(df):
                continue
            start_idx = full_index[0]
            end_time = df.iloc[start_idx + 1]['timestamp']
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
            coords = normalize_joints(coords)
            sequence.append(coords)

        sequence = np.stack(sequence) if sequence else np.zeros((1, NUM_JOINTS, CHANNELS))
        if sequence.shape[0] < FRAME_LEN:
            pad = np.zeros((FRAME_LEN - sequence.shape[0], NUM_JOINTS, CHANNELS))
            sequence = np.concatenate((sequence, pad), axis=0)
        else:
            sequence = sequence[:FRAME_LEN]

        sequence = np.transpose(sequence, (2, 0, 1))
        tensor = torch.tensor(sequence, dtype=torch.float32)

        torch.save({
            "data": tensor,
            "label": label
        }, os.path.join(output_dir, f"{folder_name}.pt"))

        print(f"  ✅ Saved tensor for {folder_name}")
    except Exception as e:
        print(f"  ❌ Error processing {folder_name}: {e}")

for folder in sorted(os.listdir(json_root_dir)):
    if os.path.isdir(os.path.join(json_root_dir, folder)):
        process_participant(folder)
