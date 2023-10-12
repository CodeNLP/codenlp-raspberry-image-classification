import time

t0 = time.perf_counter()
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification
from PIL import Image
import requests
print(f"Import: {time.perf_counter()-t0:0.2f}")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

t0 = time.perf_counter()
feature_extractor = MobileViTFeatureExtractor.from_pretrained("apple/mobilevit-small")
model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
print(f"Model: {time.perf_counter()-t0:0.2f}")

t0 = time.perf_counter()
inputs = feature_extractor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits

# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
print(f"Prediction: {time.perf_counter()-t0:0.2f}")
