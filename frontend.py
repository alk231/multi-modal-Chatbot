import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="RAG Multimodal Chatbot", page_icon="🤖", layout="wide")

# ----- Session State -----
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "sender": "bot",
            "text": "Hello! Upload files, images, audio, webpages, or YouTube links. Then ask anything from them.",
        }
    ]

# ----- Sidebar Uploads -----
st.sidebar.title("Upload Data to Knowledge Base")

# File upload (PDF, TXT, AUDIO, DOCX)
file = st.sidebar.file_uploader(
    "Upload File", type=["pdf", "txt", "mp3", "m4a", "wav", "docx"]
)
if file:
    with st.spinner("Uploading..."):
        res = requests.post(f"{API_BASE}/upload", files={"file": file})
        msg = res.json()
        st.sidebar.success(msg.get("message", "Uploaded."))

# Image Upload
image = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if image:
    with st.spinner("Uploading image..."):
        res = requests.post(f"{API_BASE}/upload", files={"file": image})
        msg = res.json()
        st.sidebar.success(msg.get("message", "Image ready for VQA."))

# URL Input
url = st.sidebar.text_input("Add Webpage URL")
if st.sidebar.button("Add URL") and url:
    with st.spinner("Processing URL..."):
        res = requests.post(f"{API_BASE}/upload", data={"url": url})
        msg = res.json()
        st.sidebar.success(msg.get("message", "Webpage added."))

# YouTube Input
yt = st.sidebar.text_input("Add YouTube Link")
if st.sidebar.button("Fetch Transcript") and yt:
    with st.spinner("Getting transcript..."):
        res = requests.post(f"{API_BASE}/youtube", data={"youtube_url": yt})
        msg = res.json()
        st.sidebar.success(msg.get("message", "Transcript added."))

st.sidebar.markdown("---")
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

# ----- Chat Display -----
st.write("## 🤖 Chat")

for msg in st.session_state.messages:
    align = "flex-end" if msg["sender"] == "user" else "flex-start"
    bubble_bg = "#007BFF" if msg["sender"] == "user" else "#FFFFFF"
    text_color = "white" if msg["sender"] == "user" else "black"

    st.markdown(
        f"""
        <div style="display:flex; justify-content:{align}; margin:8px 0;">
            <div style="
                background:{bubble_bg};
                color:{text_color};
                padding:12px 16px;
                border-radius:18px;
                max-width:70%;
                font-size:16px;
                line-height:1.4;
                ">
                {msg["text"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----- Chat Input -----
prompt = st.chat_input("Type your message...")
if prompt:
    st.session_state.messages.append({"sender": "user", "text": prompt})

    with st.spinner("Thinking..."):
        try:
            res = requests.post(f"{API_BASE}/query", json={"question": prompt}).json()
            reply = res.get("answer", "No response.")
        except:
            reply = "Server error."

    st.session_state.messages.append({"sender": "bot", "text": reply})
    st.rerun()
