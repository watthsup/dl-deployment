from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.structure import InferResponseModel
from app.inferencer import ModelInferencer
from app.config import api_config

import time
import cv2
import numpy as np

app = FastAPI(title = "Inferencing API Gateway", version="0.1", docs_url= api_config['api_prefix'] + '/docs')
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["GET", "POST"],
    allow_headers = ["*"],
)

if api_config['inference_engine'] == 'triton':
    model_url = f"{api_config['model_ip']}:{api_config['model_port']}"
    inferencer = ModelInferencer(model_name=api_config['model_name'], engine=api_config['inference_engine'], url=model_url)
if api_config['inference_engine'] == 'onnxruntime' or api_config['inference_engine'] == 'onnxruntime-gpu':    
    inferencer = ModelInferencer(model_name=api_config['model_name'], engine=api_config['inference_engine'])

@app.post('/models/infer', tags=["inferencing"], response_model=InferResponseModel)
def inference_request(file: UploadFile = File(...)):
    try:
        content = file.file.read()
        np_array = np.fromstring(content, np.uint8)
        inputImage = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        start = time.time()
        predResult = inferencer.infer(inputImage)
        end = time.time()
        endpointResponse = {}
        endpointResponse['process_time'] = end - start
        endpointResponse['top10_pred'] = [pred[0] for pred in predResult]
        endpointResponse['confidence'] = [round(float(pred[1]),4) for pred in predResult]
        endpointResponse['status'] = 'INFER_SUCCESS'

        endpointResponse = InferResponseModel(**endpointResponse)

        return endpointResponse
    except Exception as e:
        print(e)
        return InferResponseModel(status='ERR_INTERNAL_SERVER_ERROR') 
