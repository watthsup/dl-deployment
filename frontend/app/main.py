import streamlit as st
import base64
import requests
from PIL import Image
import numpy as np

st.markdown('<h1 style="color:white;">Image Classification Web App demo</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="color:white;">Predict up to 1000 classes based on ImageNet categories</p>', unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('content/bg.gif')

upload = st.file_uploader('Insert image for classification', type=['png','jpg'])
c1, c2 = st.columns(2)
if upload is not None:
    files = {"file": upload.getvalue()}
    img = Image.open(upload)
    img = np.asarray(img)
    c1.header('Input Image')
    c1.image(img)

    response = requests.post(f"http://backend:8000/models/infer", files=files)
    res = response.json()
    c2.header('Top 10 Prediction')
    pred = res['top10_pred']
    conf = res['confidence']
    text = ""
    for idx in range(len(pred)):
        text += str(idx+1) + ". " + f"**:green[{pred[idx]}]**" + " | Confidence : " + f"**:orange[{str(conf[idx])}]**" + "\n"
    c2.write(text)
