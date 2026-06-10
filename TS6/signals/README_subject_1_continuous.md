# README: Subject 1 Continuous Gravity-Axis Signal

This directory contains the reconstructed continuous time-series accelerometer signal for **Subject 1** extracted from the **UCI Human Activity Recognition (HAR) Using Smartphones Dataset**.

## File Name
* **File:** [subject_1_continuous.csv](file:///Users/julian/Downloads/human+activity+recognition+using+smartphones/subject_1_continuous.csv)
* **Format:** Comma-Separated Values (CSV)

---

## Data Specifications

| Metric | Value |
| :--- | :--- |
| **Source Subject** | Subject 1 (Volunteer 1 of the study) |
| **Sensor Source** | Smartphone Embedded Triaxial Accelerometer |
| **Signal Axis** | X-Axis (aligned with the **gravity axis** for waist-mounted position) |
| **Sampling Rate** | **50 Hz** (50 samples per second) |
| **Total Samples** | 22,272 samples |
| **Duration** | **445.44 seconds** (~7 minutes, 25.44 seconds) |

---

## Dataset Source & Context
The raw data comes from the **UCI Human Activity Recognition Using Smartphones Dataset**. 
* **Methodology:** Experiments were carried out with 30 volunteers performing six basic activities while wearing a waist-mounted Samsung Galaxy S II smartphone.
* **Original Format:** The original dataset preprocessed sensor signals into windowed frames of 128 readings (2.56 seconds) with a 50% overlap (64 samples).

---

## Reconstruction Methodology
This continuous time-series was reconstructed from the original train file `UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt`:
1. Windows corresponding to **Subject 1** were extracted sequentially (347 total windows).
2. The windows were stitched together by taking the first **64 samples** of each overlapping window.
3. For the **final window**, all 128 samples were taken to complete the signal.
4. Activity labels from `y_train.txt` and activity mappings from `activity_labels.txt` were dynamically joined to provide context for each sample point.

---

## CSV Column Details

| Column | Data Type | Description |
| :--- | :--- | :--- |
| `time_step` | Integer | Sequential index of the sample (ranges from `0` to `22271`). At 50 Hz, `time_step / 50` yields the timestamp in seconds. |
| `subject_id` | Integer | Identifier of the subject (always `1` for this file). |
| `activity_id` | Integer | Activity code representing the physical activity of the subject (range `1` to `6`). |
| `activity_name` | String | Human-readable activity name: `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`. |
| `total_acc_x` | Float | Total acceleration (body acceleration + gravity component) along the X-axis (gravity axis), measured in standard gravity units ($g$, where $1g \approx 9.80665 \text{ m/s}^2$). |
