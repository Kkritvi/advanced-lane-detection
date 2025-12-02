@"
# Advanced Lane Detection (BEV + LDW + Webcam)

A complete lane detection pipeline using classical computer vision techniques (no deep learning), featuring HLS color masking, ROI extraction, bird's-eye perspective transform, sliding-window lane pixel extraction, polynomial fitting, temporal smoothing, lane departure warning (LDW), and real-time webcam mode using DirectShow backend.

## 📌 Demo
(Screenshot: snapshot_output.png)

## 🚀 Features
- HLS color thresholding for robust white & yellow line detection  
- Region of Interest masking  
- Bird’s-Eye View (BEV / perspective transform)  
- Sliding Window lane detection  
- 2nd-degree polynomial lane curve fitting  
- Temporal smoothing using history buffer  
- Lane Departure Warning (LDW) with offset in meters and pixels  
- Real-time webcam support (`--webcam`)  
- Video input support (`--video`)  

## ▶️ Running the Project
### Install dependencies:


### Run on video:


### Run on webcam:


### Controls:
- `q` → quit  
- `s` → save a snapshot (`snapshot_output.png`)

## 📁 Files
- `main.py` — main script  
- `snapshot_output.png` — sample output (add your own)  
- `video.mp4` — optional sample input  
- `output.mp4` — optional processed output  
- `requirements.txt` — dependencies  
- `LICENSE` — MIT License  

## 💡 Why this project is interesting
This project demonstrates:
- Geometric transforms  
- Image processing  
- Real-time computer vision  
- Lane estimation & tracking  
- ADAS concepts (Lane Departure Warning)  

Suitable for robotics, CV, ECE, and autonomous driving research.

## 📜 License
MIT License  
"@ | Out-File -FilePath README.md -Encoding utf8
