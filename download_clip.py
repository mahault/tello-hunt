"""
Pre-download CLIP model for offline use.

Run this script while connected to the internet (before connecting to Tello WiFi).

Usage:
    python download_clip.py
"""

import torch

def main():
    print("=" * 50)
    print("CLIP Model Pre-Download")
    print("=" * 50)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("No GPU detected, will use CPU")

    print("\nDownloading CLIP model (openai/clip-vit-base-patch32)...")
    print("This may take a few minutes on first run.\n")

    from transformers import CLIPProcessor, CLIPModel

    # Download and cache the model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("\nModel downloaded and cached!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test that it works
    print("\nTesting model with a dummy image...")
    import numpy as np
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    inputs = processor(images=dummy_image, return_tensors="pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    print(f"Output shape: {features.shape}")
    print(f"Device used: {device}")
    print("\nSUCCESS! Model is ready for offline use.")
    print("You can now connect to Tello WiFi and run:")
    print("  python person_hunter_pomdp.py --ground-test")


if __name__ == "__main__":
    main()
