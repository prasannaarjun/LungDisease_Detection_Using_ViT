import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

def load_model(path, num_labels):
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=num_labels
    )
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_labels)
    model.load_state_dict(torch.load(path, map_location=device,weights_only=True))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image: Image.Image):
    return feature_extractor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

def predict_disease(stage1_model, stage2_model, image: Image.Image):
    inputs = preprocess_image(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_stage1 = stage1_model(inputs).logits
        pred_stage1 = torch.argmax(logits_stage1, dim=1).item()

    stage1_labels = ['Corona Virus Disease', 'Normal', 'Pneumonia', 'Tuberculosis']

    result = stage1_labels[pred_stage1]

    if result == 'Pneumonia':
        with torch.no_grad():
            logits_stage2 = stage2_model(inputs).logits
            pred_stage2 = torch.argmax(logits_stage2, dim=1).item()
        pneumonia_type = ['Bacterial Pneumonia', 'Viral Pneumonia'][pred_stage2]
        return f" {pneumonia_type}"

    return result
