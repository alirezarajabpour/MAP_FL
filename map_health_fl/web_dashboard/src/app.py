import os
import torch
from flask import Flask, request, render_template, jsonify
from PIL import Image
import torchvision.transforms as transforms
import io
import base64

# Assuming the same model definition is available
# In a real project, this would be in a shared library
from models import Net

app = Flask(__name__)

# --- Load Model ---
MODEL_PATH = "/model/global_model.pth"
model = Net(num_classes=21) # Make sure num_classes matches
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("Model loaded successfully!")
else:
    print("WARNING: Model file not found.")

# --- Image Transformations ---
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5]) # Adjust for your dataset
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400
    
    file = request.files['file']
    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Transform the image and add batch dimension
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs.data, 1)
            prediction_idx = predicted.item()
        
        # In a real app, you would map this index back to a class name
        class_name = f"Predicted Class: {prediction_idx}"

        # Prepare image to be displayed in HTML
        img_str = base64.b64encode(img_bytes).decode('utf-8')

        return jsonify({'prediction': class_name, 'image': img_str})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)