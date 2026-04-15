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

## 1) Install Dependencies

```powershell
cd "e:\dl\image_captioning"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Prepare Data + Extract Features

```powershell
python data_prep.py --dataset_dir "e:\dl\archive (1)" --output_dir "e:\dl\image_captioning\artifacts"
```

This will:

- Clean captions and add `startseq` / `endseq`
- Build tokenizer
- Split images into train/val
- Extract ResNet50 features for all images

## 3) Train the Model

```powershell
python train.py --artifacts_dir "e:\dl\image_captioning\artifacts" --checkpoints_dir "e:\dl\image_captioning\checkpoints" --epochs 20 --batch_size 128
```

Expected runtime on CPU can be long (hours).

## 4) Launch Web UI (Streamlit)

```powershell
streamlit run app.py
```

In the app:

- Use **Custom CNN+LSTM** if you finished training.
- Use **BLIP fallback** for immediate quality captions.
- Use **Auto (Custom -> BLIP)** for best convenience.

## 5) Launch Backend API + Frontend

Start API:

```powershell
uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload
```

Start frontend (new terminal):

```powershell
cd "e:\dl\image_captioning\frontend"
python -m http.server 5500
```

Open:

- `http://127.0.0.1:5500`
- API endpoint is prefilled as `http://127.0.0.1:8000/caption`

The frontend sends the uploaded image to FastAPI and displays the generated caption.

## 6) React Frontend (Aesthetic UI)

Start FastAPI backend first:

```powershell
cd "e:\dl\image_captioning"
uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload
```

Run the React app in a second terminal:

```powershell
cd "e:\dl\image_captioning\frontend-react"
npm install
npm run dev
```

Open:

- `http://127.0.0.1:5173`

This React UI is already connected to `http://127.0.0.1:8000/caption` by default and supports all three modes: `auto`, `custom`, and `blip`.
