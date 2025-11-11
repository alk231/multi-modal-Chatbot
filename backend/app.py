from flask import Flask, request, jsonify
from flask_cors import CORS
import os, math
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader, Docx2txtLoader
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from pydub import AudioSegment
from langchain_core.documents import Document

import whisper

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Whisper model
whisper_model = whisper.load_model("small")

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector DB
vector_store = None
VECTOR_STORE_PATH = "faiss_index"
last_image_path = None

# Load existing memory
if os.path.exists(VECTOR_STORE_PATH):
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)


from langchain_core.documents import Document


def add_to_vectorstore(text, source):
    global vector_store

    # Reset index on every new upload
    vector_store = None
    if os.path.exists(VECTOR_STORE_PATH):
        import shutil

        shutil.rmtree(VECTOR_STORE_PATH)

    chunks = text_splitter.split_text(text)
    documents = [
        Document(page_content=chunk, metadata={"source": source}) for chunk in chunks
    ]

    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local(VECTOR_STORE_PATH)


def extract_text_from_pdf(path):
    loader = PyPDFLoader(path)
    text = "\n".join(p.page_content for p in loader.load())
    add_to_vectorstore(text, path)
    return text


def extract_text_from_txt(path):
    loader = TextLoader(path)
    text = "\n".join(p.page_content for p in loader.load())
    add_to_vectorstore(text, path)
    return text


def extract_text_from_docx(path):
    loader = Docx2txtLoader(path)
    docs = loader.load()  # returns: [Document(page_content="full text...")]
    text = docs[0].page_content  # extract the text correctly
    add_to_vectorstore(text, path)
    return text


def extract_text_from_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    text = "\n".join(d.page_content for d in docs)
    add_to_vectorstore(text, url)
    return text


def extract_text_from_audio(path):
    ext = os.path.splitext(path)[1].lower()
    if ext != ".wav":
        sound = AudioSegment.from_file(path)
        path = os.path.splitext(path)[0] + ".wav"
        sound.export(path, format="wav")

    audio = AudioSegment.from_wav(path)
    chunk_ms = 60_000
    text_full = ""

    for i in range(math.ceil(len(audio) / chunk_ms)):
        seg = audio[i * chunk_ms : (i + 1) * chunk_ms]
        temp = f"{path}_chunk.wav"
        seg.export(temp, format="wav")
        text_full += " " + whisper_model.transcribe(temp)["text"]
        os.remove(temp)

    add_to_vectorstore(text_full.strip(), path)
    return text_full.strip()


def get_youtube_text(url):
    if "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    else:
        video_id = url.split("youtu.be/")[1].split("?")[0]

    try:
        api = YouTubeTranscriptApi()
        snippets = api.fetch(video_id, languages=["en", "hi"])
        text = " ".join(s.text for s in snippets)
        add_to_vectorstore(text, url)
        return text

    except Exception:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcripts.find_generated_transcript(["en", "hi"])
        parts = transcript.fetch()
        text = " ".join(p["text"] for p in parts)
        add_to_vectorstore(text, url)
        return text


@app.route("/upload", methods=["POST"])
def upload():
    global last_image_path

    if "url" in request.form:
        extract_text_from_url(request.form["url"])
        return jsonify({"success": True, "message": "🌐 Website stored."})

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})

    file = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    ext = os.path.splitext(path)[1].lower()
    file.save(path)

    if ext == ".pdf": extract_text_from_pdf(path)
    elif ext == ".txt": extract_text_from_txt(path)
    elif ext == ".docx": extract_text_from_docx(path)
    elif ext in [".mp3", ".m4a", ".wav"]: extract_text_from_audio(path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        last_image_path = path
        return jsonify({"success": True, "message": "🖼️ Image stored for visual reasoning."})
    else:
        return jsonify({"success": False, "error": "Unsupported file type"})

    return jsonify({"success": True, "message": "✅ Stored successfully."})


@app.route("/youtube", methods=["POST"])
def youtube():
    text = get_youtube_text(request.form["youtube_url"])
    return jsonify({"success": True, "message": "🎬 Transcript stored.", "text": text})


@app.route("/query", methods=["POST"])
def query():
    global last_image_path

    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"success": False, "answer": "Ask a question first."})

    if vector_store is None:
        return jsonify({"success": False, "answer": "No data uploaded yet."})

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)
    context = "\n".join(d.page_content for d in docs)

    llm = ChatGroq(model="openai/gpt-oss-120b")
    parser = StrOutputParser()

    # Vision mode
    if last_image_path:
        with open(last_image_path, "rb") as img:
            result = llm.invoke({"input": question + "\n\nContext:\n" + context, "image": img.read()})
        return jsonify({"success": True, "answer": result})

    # RAG mode
    prompt = PromptTemplate(
        template=(
            "Use ONLY the context to answer.\n"
            "If answer is not in context, reply: 'I don't know'.\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        ),
        input_variables=["context", "question"],
    )

    chain = prompt | llm | parser
    answer = chain.invoke({"context": context, "question": question})

    return jsonify({"success": True, "answer": answer})


@app.route("/reset", methods=["POST"])
def reset():
    global vector_store
    vector_store = None
    if os.path.exists(VECTOR_STORE_PATH):
        import shutil
        shutil.rmtree(VECTOR_STORE_PATH)
    return jsonify({"success": True, "message": "🧹 Memory cleared."})


if __name__ == "__main__":
    app.run(debug=True, port=8000)
