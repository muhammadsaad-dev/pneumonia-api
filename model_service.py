import io
import torch 
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class PneumoniaModel:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.model_path = model_path
        self.img_size = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.model = self.load_model()

    def load_model(self):
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),      # Drop 50% of neurons to prevent overfitting
            nn.Linear(num_ftrs, 2)
        )
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()
        return model
    
    def transform(self, image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        IMG_SIZE = self.img_size
        preprocessor = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)), 
            transforms.ToTensor(), 
            transforms.Normalize(self.mean, self.std)
        ])
        return preprocessor(image)
    
    def predict(self, image_bytes: bytes):
    # 1. Transform
        image_tensor = self.transform(image_bytes).unsqueeze(0).to(self.device)

    # 2. Inference
        with torch.inference_mode():
            outputs = self.model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        
        prob_dict = {self.classes[i]: float(probs[0][i].item()) for i in range(len(self.classes))}

        return prob_dict


    

        