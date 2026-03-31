import cv2
import torch
import numpy as np
from torchvision import transforms

def preprocess_image(image):
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std = [0.229, 0.224, 0.225])
  image = cv2.resize(image, (224,224))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = image.astype(np.float32)/255
  image = np.transpose(image, (2,0,1))
  image = torch.from_numpy(image)
  image = normalize(image)

  image = image.unsqueeze(0)

  return image