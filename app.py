import streamlit as st
from PIL import Image
from models.caption import generate_caption
from models.qna import ask_question

st.set_page_config(page_title="🖼️ Image Caption & QnA", layout="centered")
st.title("🖼️ Visual Caption & 💬 QnA")

if "caption" not in st.session_state:
    st.session_state.caption = ""

# Image upload and captioning section
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        st.session_state.caption = generate_caption(image)

    st.success("Caption generated!")
    st.markdown(f"**Caption:** _{st.session_state.caption}_")

# QnA section - persistent while caption exists
if st.session_state.caption:
    question = st.text_input("Ask a question about the image caption")
    if question:
        with st.spinner("Thinking..."):
            answer = ask_question(st.session_state.caption, question)
        st.markdown(f"**Answer:** {answer}")
