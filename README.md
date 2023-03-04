# Image Classification Deployment

This project is a full-stack application capable of performing image classification in 1000 categories from ImageNet. The application uses a pre-trained ResNet18 model on ImageNet and contains three components:

1. Front-end: A Streamlit-based web applitcation to serve client interaction.
2. Back-end: A FastAPI-based API app that processes the user's request and sends it to the inference server.
3. Inference server: A Triton Docker container that serves the deep learning model and enhances concurrent usage.

The purpose of this project is to demonstrate deep learning deployment architecture and serve as a portfolio piece.

## Key Features

- End-to-end application for easy use
- Low latency and high throughput due to the use of Triton inference server
- Scalable and portable deployment with Docker Compose
## Programming Language and Libraries

- Python
- OpenCV
- PyTorch
- ONNX
- Triton
- NumPy
- FastAPI
- Streamlit
- Docker Compose

## Usage

To use the application, follow the steps below:

1. Clone the repository to your local machine.
2. Navigate to the root directory of the project.
3. Run `docker-compose up` to build and start the application.
4. Access the Streamlit frontend by navigating to `http://localhost:8501`.

## Preview
![ezgif com-video-to-gif (3)](https://user-images.githubusercontent.com/121663706/222917652-8e9f0775-1ca3-4f34-a819-248d69460020.gif)
