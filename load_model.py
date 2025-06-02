import torch
import pickle
from Models.cnn14 import SpeechCommandNet

# Load the label encoder
with open('Models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(n_classes=12, device='cpu'):
    model = SpeechCommandNet(n_classes=n_classes)
    model = model.to(device)
    model.eval()
    return model

model = get_model(n_classes=len(le.classes_), device=device)

try:
    checkpoint = torch.load('Models/best_model.pth', map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

model.eval()

def get_predictions(audio_batch):
    with torch.no_grad():
        predictions = model(audio_batch)
        return predictions.argmax(1).cpu().numpy()

class_names = le.classes_ 