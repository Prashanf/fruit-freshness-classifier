import os
from fastapi import FastAPI, File, UploadFile
import uvicorn
import torch
import cv2
import numpy as np
import json
from src.utils import preprocess_image
from src.models import TreeCnn, TreeCnn2

#initialsing app
app = FastAPI()

#Loading classes
with open("classes.json") as f:
    classes = json.load(f)


#Loading the model
device = torch.device("cpu")
model = TreeCnn2()
model.load_state_dict(torch.load("models/treeCnn2.pth", map_location=device))
model.eval()

@app.post("/predict")
async def predict(file:UploadFile=File()):
    #readig image
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    image = preprocess_image(image)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs,dim=1)
        conf, idx = torch.max(probs, dim=1)

    return{
        "class":classes[idx.item()],
        "confidence": float(conf.item())
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port)