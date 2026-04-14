# Segmentation Model Service — Demo Instructions

## 1. Clone the repository
git clone <your-repo-link>
cd <your-repo-folder>

## 2. Set up environment
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
# OR
venv\Scripts\activate      # (Windows)

pip install -r requirements.txt

## 3. Run dataset preparation
python prepare_dataset.py

This will:
- download the aerial dataset
- generate pixel masks
- create train / validation / test splits

## 4. Train the segmentation model
python train_segmentation.py

This will:
- train the segmentation model
- print IoU and Dice scores
- generate:
  - loss_curve.png
  - predictions/ (images, ground truth, predicted masks)
  - house_segmentation_model.pth

## 5. Run the API (original Lab 1 service)
python app.py

Server runs at:
http://localhost:5000

## 6. Test endpoints

Health check:
curl http://localhost:5000/health

Prediction endpoint:
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"This lab works well\"}"

## 7. Docker (optional demo)
docker build -t model-service .
docker run -p 5000:5000 model-service

## 8. CI/CD
Push to the main branch:
git push

This triggers:
- tests
- Docker build
- Docker image push to Docker Hub

Check results in:
GitHub → Actions tab

## Notes
- Segmentation model is trained separately from the API
- Mini dataset is used for demonstration purposes
- Small dataset may cause overfitting (expected)