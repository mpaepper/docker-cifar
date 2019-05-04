import os, io, json, tarfile, glob, time, logging, base64, boto3, PIL, torch
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_model():
    model_dir = 'model'
    classes = open(f'{model_dir}/classes', 'r').read().splitlines()
    logger.info(f'Classes are {classes}')    
    model_path = f'{model_dir}/simplecifar_jit.pth'
    logger.info(f'Model path is {model_path}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(model_path, map_location=device)
    return model.eval(), classes

model, classes = load_model()

def predict(model, classes, image_tensor):
    """Predicts the class of an image_tensor."""

    start_time = time.time()
    predict_values = model(image_tensor)
    logger.info("Inference time: {} seconds".format(time.time() - start_time))
    softmaxed = F.softmax(predict_values, dim=1)
    probability_tensor, index = torch.max(softmaxed, dim=1)
    prediction = classes[index]
    probability = "{:1.2f}".format(probability_tensor.item())
    logger.info(f'Predicted class is {prediction} with a probability of {probability}')
    return {'class': prediction, 'probability': probability}

preprocess_pipeline = transforms.Compose([
    transforms.Resize(50),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
    
def image_to_tensor(img):
    """Transforms the posted image to a PyTorch Tensor."""
    
    img_tensor = preprocess_pipeline(img)
    img_tensor = img_tensor.unsqueeze(0) # 3d to 4d for batch
    return img_tensor
    
def inference(img):
    """The main inference function which gets passed an image to classify"""

    image_tensor = image_to_tensor(img)
    response = predict(model, classes, image_tensor)
    return {
        "statusCode": 200,
        "body": json.dumps(response)
    }
