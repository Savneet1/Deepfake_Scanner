import os
import argparse
import torch
from torchvision import transforms
from PIL import Image

from train_model import create_model  # make sure this import works

DEVICE = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model(model_path):
    model = create_model()
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

def predict_folder(model, folder, label_name):
    images = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    correct = 0
    total = 0

    for path in images:
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            pred = torch.argmax(logits, dim=1).item()  # 0=Real, 1=Fake

        total += 1
        is_correct = (pred == label_name)
        correct += int(is_correct)

        print(f"{path} -> {'Real' if pred == 0 else 'Fake'}")

    if total > 0:
        acc = 100.0 * correct / total
        print(f"\nFolder: {folder}")
        print(f"Expected label: {'Real' if label_name == 0 else 'Fake'}")
        print(f"Correct: {correct}/{total} ({acc:.2f}% correct)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/checkpoints/best_model.pt")
    parser.add_argument("--real_dir", required=True, help="Path to test real images")
    parser.add_argument("--fake_dir", required=True, help="Path to test fake images")
    args = parser.parse_args()

    model = load_model(args.model)

    # Real label = 0, Fake label = 1
    predict_folder(model, args.real_dir, 0)
    predict_folder(model, args.fake_dir, 1)
