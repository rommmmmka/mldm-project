import mne
import numpy as np
import torch


def load_dataset(channel_name="F3-A2"):
    subjects = []

    old_log_level = mne.set_log_level("WARNING", return_old_level=True)
    for subject in range(1, 11):
        print(f"Loading subject {subject}")
        raw_edf = mne.io.read_raw_edf(
            f"isruc_s3/{subject}/{subject}.edf", preload=True, include=[channel_name]
        )
        epochs = mne.make_fixed_length_epochs(raw_edf, duration=30, preload=True)
        tensor = torch.tensor(epochs.get_data(), dtype=torch.float32).to("cuda")
        subjects.append(tensor)
    mne.set_log_level(old_log_level)

    return subjects


def load_annotations(annotator=1):
    annotations = []

    for subject in range(1, 11):
        with open(
            f"isruc_s3/{subject}/{subject}_{annotator}.txt", mode="r", encoding="utf-8"
        ) as file:
            subject_annotations = list(map(int, file.read().strip().split("\n")))
            subject_annotations_np = np.array(subject_annotations)
            subject_annotations_np[subject_annotations_np == 5] = 4
            tensor = torch.tensor(subject_annotations_np).to("cuda")
            annotations.append(tensor)

    return annotations
