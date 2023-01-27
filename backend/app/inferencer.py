import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import onnxruntime as rt
import tritonclient.grpc as client
import torch
from scipy.special import softmax
import tritonclient.grpc as client

class ModelInferencer():
    def __init__(self, model_name:str, engine:str = 'triton', url:str=None):
        self.engine = engine
        self.model_name = model_name
        self.url = url
        with open('./model_repository/labels.txt') as f:
            classes = [line.strip() for line in f.readlines()]
        self.labels = classes
        if self.engine == 'onnxruntime' or self.engine == 'onnxruntime-gpu':
            sessOpt = rt.SessionOptions()
            sessOpt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            sessOpt.add_session_config_entry("session.set_denormal_as_zero", "1")
            
            if self.engine == 'onnxruntime':
                providers=['CPUExecutionProvider']
            else:
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']

            self.inferencer = rt.InferenceSession(f'model_repository/{self.model_name}/1/model.onnx', sessOpt, providers=providers)
        elif self.engine == 'triton':
            self.tritonConnector = client.InferenceServerClient(url=self.url)
    
    def infer(self, image_numpy):
        if self.engine == 'onnxruntime' or self.engine == 'onnxruntime-gpu':
            input_batch = self.preprocess(image_numpy)
            onnx_output = self.inferencer.run(None, {'input' : input_batch})
            onnx_output = torch.FloatTensor(np.array(onnx_output)).reshape(1,-1)
            results = self.postprocess(onnx_output)
        elif self.engine == 'triton':
            input_batch = self.preprocess(image_numpy)
            input0 = client.InferInput('input', (1, 3, 224, 224), "FP16")
            input0.set_data_from_numpy(input_batch)
            output = self.tritonConnector.infer(model_name=self.model_name, inputs=[input0])
            logits = output.as_numpy('output') 
            results = self.postprocess(logits)
        return results
        
    def preprocess(self, image_numpy):
        transform = A.Compose(
                [
                    A.Resize(height=224, width=224),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255., 
                        p=1.0
                    ),
                    ToTensorV2(),
                ])
        transform = transform(image=image_numpy)
        image = transform["image"]
        input_batch = image.unsqueeze(0).numpy()
        
        if self.engine == 'onnxruntime' or self.engine == 'onnxruntime-gpu':
            input_batch = input_batch.astype(np.float32)
        if self.engine == 'triton':
            input_batch = input_batch.astype(np.float16)
        return input_batch
        
    def postprocess(self, output, topK=10):
        if self.engine == 'onnxruntime' or self.engine == 'onnxruntime-gpu':
            onnx_output = output
            onnx_output = torch.FloatTensor(np.array(onnx_output)).reshape(1,-1)
            prob = torch.nn.functional.softmax(onnx_output, dim=1)[0] * 100
            _, indices = torch.sort(prob, descending=True)
            return [(self.labels[idx], prob[idx].item()) for idx in indices[:topK]]
        elif self.engine == 'triton':
            triton_output = output
            logits = np.asarray(triton_output, dtype=np.float32)
            probs = softmax(logits)
            probs = np.sort(probs).reshape(-1)[::-1] * 100
            indices = np.argsort(logits).reshape(-1)[::-1]
            zipped = list(zip(indices,probs))
            return [(self.labels[elem[0]], elem[1]) for elem in zipped[:topK]]
            
