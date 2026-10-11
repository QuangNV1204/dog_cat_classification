import io

import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from PIL import Image


LABELS = ["Mèo", "Chó"]
MODEL_PATH = "best_nin.pth"


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def build_model():
    return nn.Sequential(
        nin_block(3, 96, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nin_block(96, 192, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nin_block(192, 384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Dropout(0.5),
        nn.Conv2d(384, 2, kernel_size=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )


@st.cache_resource
def load_model():
    model = build_model()
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict(model, img: Image.Image):
    x = preprocess(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    return LABELS[idx], float(probs[idx]), [float(p) for p in probs]


st.title("🐱🐶 Dog-Cat Classifier (NIN)")

uploaded = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    img = Image.open(io.BytesIO(uploaded.read()))
    st.image(img, caption="Ảnh input", use_container_width=True)

    try:
        model = load_model()
    except FileNotFoundError:
        st.error(f"Không thấy {MODEL_PATH}. Thả file best_nin.pth vào cùng thư mục với app.py.")
        st.stop()
    except RuntimeError as e:
        st.error("Load được file nhưng lệch kiến trúc (size mismatch). "
                 "Kiểm tra lại nin_block có khớp lúc train không.")
        st.exception(e)
        st.stop()

    if st.button("Dự đoán", type="primary"):
        label, conf, all_probs = predict(model, img)
        st.subheader(f"Kết quả: **{label}** — {conf:.1%}")
        st.progress(conf)
        st.write({LABELS[i]: f"{p:.1%}" for i, p in enumerate(all_probs)})
else:
    st.info("Tải 1 ảnh mèo/chó để thử.")
