# Guide: Running MOS Evaluation for Hindi Audiobooks

This guide explains how to set up and run a Mean Opinion Score (MOS) test using the **ReSEval** framework.

---

## 1. Environment Setup
Run these commands in your terminal to prepare the workspace:

```bash
# 1. Install ReSEval
pip install reseval

# 2. Install MySQL (Required for local testing)
sudo apt-get update
sudo apt-get install -y mysql-server
sudo service mysql start

# 3. Create a Local Database User
# Change 'reseval_user' and 'reseval_pass' if you prefer
sudo mysql -e "CREATE USER IF NOT EXISTS 'reseval_user'@'localhost' IDENTIFIED BY 'reseval_pass';"
sudo mysql -e "GRANT ALL PRIVILEGES ON * . * TO 'reseval_user'@'localhost';"
sudo mysql -e "FLUSH PRIVILEGES;"

# 4. Save Credentials for ReSEval
python -m reseval.credentials --mysql_local_user reseval_user --mysql_local_password reseval_pass
```

---

## 2. Where to Put Your Audio
The directory structure is already created in `mos_test_data/`.

1.  **Generate your audio** for 4 different models (or conditions).
2.  **Organize them** as follows:
    ```text
    mos_test_data/
    ├── model_1/
    │   ├── utterance_01.wav
    │   ├── utterance_02.wav
    ├── model_2/
    │   ├── utterance_01.wav
    │   ├── utterance_02.wav
    ├── ... (and so on)
    ```
3.  **Crucial Rule:** The filenames must be **identical** across all folders. For example, if `story_clip_1.wav` exists in `model_1`, it should also exist in `model_2`, `model_3`, and `model_4`. ReSEval uses this to let different participants compare the same text spoken by different models.

---

## 3. Customizing the Configurations
You now have two specific configuration files:
*   `nmos_config.yaml`: Used to measure **Naturalness** (how human-like the voice is).
*   `emos_config.yaml`: Used to measure **Emotionality** (how expressive the voice is).

Open either file to adjust these key parameters:

| Parameter | What it does |
| :--- | :--- |
| `name` | Unique ID. **Must be different** for NMOS vs EMOS to avoid data collisions. |
| `participants` | How many unique crowdworkers to hire. |
| `samples_per_participant` | How many clips each person rates. |

---

## 4. Running the Test

### **Phase A: Local Preview (Free)**
Test your UI for either Naturalness or Emotion:
```bash
# To test Naturalness
python -m reseval.create nmos_config.yaml mos_test_data/ --local

# To test Emotion
python -m reseval.create emos_config.yaml mos_test_data/ --local
```

### **Phase B: Production (AWS/MTurk)**
When ready for real ratings:
1.  **Add Credentials** (see section 1).
2.  **Update the YAML**: Set `storage`, `database`, and `server` to `aws` in the chosen config file.
3.  **Launch**:
    ```bash
    python -m reseval.create nmos_config.yaml mos_test_data/ --production
    ```

---

## 5. Getting Results
Download results by referencing the `name` used in your YAML:
```bash
# For Naturalness results
python -m reseval.results hindi-audiobook-nmos

# For Emotion results
python -m reseval.results hindi-audiobook-emos
```
This will create a folder with:
*   `results.csv`: All individual ratings.
*   `stats.csv`: Average scores (MOS) for each of your 4 models.
*   `mos.png`: A visualization of the results.
