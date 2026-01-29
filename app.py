import streamlit as st
import torch
from diffusers import DiffusionPipeline

# ---------------- CONFIG ----------------
MODEL_ID = "SG161222/Realistic_Vision_V2.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="AI Image Chat",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ AI Image Generator (ChatGPT Style)")
st.caption("Model: Realistic Vision | Normal vs Professional Prompt")

# ---------------- LOAD MODEL (ONCE) ----------------
@st.cache_resource
def load_model():
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    return pipe

pipe = load_model()

# ---------------- CHAT INPUT ----------------
st.subheader("Enter Prompts")

normal_prompt = st.text_input(
    "Normal Prompt",
    placeholder="e.g. A colorful butterfly sitting on a flower"
)

professional_prompt = st.text_input(
    "Professional Prompt",
    placeholder="e.g. Macro photograph, ultra-detailed, realistic lighting"
)

generate = st.button("Generate Images")

# ---------------- GENERATION ----------------
if generate and normal_prompt and professional_prompt:
    with st.spinner("Generating images..."):
        img1 = pipe(
            normal_prompt,
            num_inference_steps=25,
            guidance_scale=7.5
        ).images[0]

        img2 = pipe(
            professional_prompt,
            num_inference_steps=25,
            guidance_scale=7.5
        ).images[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Normal Prompt Output")
        st.image(img1, use_column_width=True)
    with col2:
        st.subheader("Professional Prompt Output")
        st.image(img2, use_column_width=True)
