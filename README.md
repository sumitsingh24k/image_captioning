# Image Captioning Mini-Project

This project trains an image captioning model on Flickr8k-style data using:

- **Encoder**: ResNet50 image features
- **Decoder**: LSTM caption generator (TensorFlow/Keras)
- **UI**: Streamlit app and separate Frontend + FastAPI backend

## Project Structure

```text
e:\dl\image_captioning
├── app.py
├── backend_api.py
├── caption_service.py
├── data_prep.py
├── frontend/
│   ├── index.html
│   ├── main.js
│   └── styles.css
├── train.py
├── requirements.txt
├── artifacts/
│   ├── cleaned_captions.csv
│   ├── image_features.pkl
│   ├── metadata.json
│   └── tokenizer.pkl
└── checkpoints/
    ├── best_model.keras
    ├── model.keras
    └── model.h5
```

## Quick Start (Clone & Run)

### 1) Clone & Setup Environment

```bash
git clone https://github.com/sumitsingh24k/image_captioning.git
cd image_captioning
python -m venv venv
# On Windows:
venv\Scripts\activate.ps1
# On macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare Dataset

You need a Flickr8k-style dataset with:
- `captions.txt` (columns: image, caption)
- `Images/` folder with images

**Option A: Use Your Own Dataset**
```bash
python data_prep.py --dataset_dir /path/to/your/dataset --output_dir ./artifacts
```

**Option B: Download Sample Dataset**
Visit [Flickr8k Dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip) and extract, then run above command.

This generates:
- `cleaned_captions.csv` - processed captions
- `image_features.pkl` - ResNet50 features
- `tokenizer.pkl` - vocabulary

### 3) Train the Model (Optional - Takes Hours)

```bash
python train.py --epochs 20 --batch_size 128
```

Model will be saved to `checkpoints/best_model.keras`

## 4) Run the Application

### Option A: Streamlit Web UI (Recommended)

```bash
streamlit run app.py
```

Features:
- **Custom Model**: Uses trained CNN+LSTM (if trained)
- **BLIP Fallback**: Fast captions using pre-trained BLIP model
- **Auto Mode**: Tries custom model, falls back to BLIP

### Option B: FastAPI Backend + HTML Frontend

Terminal 1 - Start Backend:
```bash
uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload
```

Terminal 2 - Start Frontend (from `frontend/` folder):
```bash
cd frontend
python -m http.server 5500
```

Open browser: `http://127.0.0.1:5500`
- Upload image → Backend generates caption using `/caption` endpoint

### Option C: React Frontend (Advanced UI)

Terminal 1 - Start Backend (same as Option B):
```bash
uvicorn backend_api:app --host 127.0.0.1 --port 8000
```

Terminal 2 - Start React Frontend:
```bash
cd frontend-react
npm install
npm run dev
```

Open browser: `http://127.0.0.1:5173`

---

## Troubleshooting

**❌ File Not Found Error**
- Make sure you ran `data_prep.py` first
- Dataset path must contain images and captions.txt

**❌ CUDA/GPU Issues**
- CPU works fine, just slower
- TensorFlow/PyTorch will auto-detect GPU if available

**❌ Missing dependencies**
- Run: `pip install -r requirements.txt` again
- Use `pip install --upgrade tensorflow torch`

**❌ Large file push issues (already fixed!)**
- Artifacts are in `.gitignore` - regenerated locally
- Never commit: `.venv/`, `checkpoints/`, artifact files
