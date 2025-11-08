from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from youtube_transcript_api import YouTubeTranscriptApi
from pydub import AudioSegment
import whisper, math, os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import Docx2txtLoader
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Whisper Model
whisper_model = whisper.load_model("small")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

last_image_path = None
vector_store = None
VECTOR_STORE_PATH = "faiss_index"

# Load existing memory
if os.path.exists(VECTOR_STORE_PATH):
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True
    )

# SPLITTER
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)


def add_to_vectorstore(text):
    global vector_store
    docs = text_splitter.split_text(text)
    if vector_store is None:
        vector_store = FAISS.from_texts(docs, embedding_model)
    else:
        vector_store.add_texts(docs)

    # ✅ Persist memory
    vector_store.save_local(VECTOR_STORE_PATH)


def extract_text_from_pdf(path):
    loader = PyPDFLoader(path)
    text = "\n".join([page.page_content for page in loader.load()])
    add_to_vectorstore(text)
    return text


def extract_text_from_txt(path):
    loader = TextLoader(path)
    text = "\n".join([page.page_content for page in loader.load()])
    add_to_vectorstore(text)
    return text


def extract_text_from_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load()  # returns list of Document objects
    text = "\n".join([d.page_content for d in docs])  # extract text properly
    add_to_vectorstore(text)
    return text

def extract_text_from_docx(path):
    loader = Docx2txtLoader(path)
    text = "\n".join(loader.load()[0].page_content)
    add_to_vectorstore(text)
    return text

def extract_text_from_audio(path):
    ext = os.path.splitext(path)[1].lower()
    if ext != ".wav":
        sound = AudioSegment.from_file(path)
        path = os.path.splitext(path)[0] + ".wav"
        sound.export(path, format="wav")

    audio = AudioSegment.from_wav(path)
    chunk_ms = 60 * 1000
    total = len(audio)
    chunks = math.ceil(total / chunk_ms)
    text_full = ""

    for i in range(chunks):
        seg = audio[i * chunk_ms : min((i + 1) * chunk_ms, total)]
        temp_path = f"{path}_chunk_{i}.wav"
        seg.export(temp_path, format="wav")
        result = whisper_model.transcribe(temp_path)
        text_full += " " + result["text"]
        os.remove(temp_path)

    add_to_vectorstore(text_full)
    return text_full.strip()


from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)


def get_youtube_text(url: str):
    # Extract video_id from URL
    if "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL format")

    try:
        # ===== Path A: v1.x API (instance methods: .list(), .fetch()) =====
        # Your inspect() earlier showed a class with .fetch() and .list().
        api = YouTubeTranscriptApi()  # v1.x style
        # Try Hindi → English; you can change order
        snippets = api.fetch(video_id, languages=["hi", "en"])
        # v1.x returns objects with `.text` (NOT dicts)
        text = " ".join(s.text for s in snippets)
        add_to_vectorstore(text)
        return text

    except AttributeError:
        # ===== Path B: v0.x API (static .list_transcripts()) =====
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcripts.find_transcript(["hi"])
        except:
            try:
                transcript = transcripts.find_transcript(["en"])
            except:
                transcript = transcripts.find_generated_transcript(["hi", "en"])
        parts = transcript.fetch()
        # v0.x returns list of dicts with ['text']
        text = " ".join(p["text"] for p in parts)
        add_to_vectorstore(text)
        return text

    except TranscriptsDisabled:
        raise RuntimeError("This video has subtitles disabled.")
    except NoTranscriptFound:
        raise RuntimeError("No subtitles available for this video.")
    except VideoUnavailable:
        raise RuntimeError("Video is unavailable.")
    except Exception as e:
        raise RuntimeError(f"YouTube transcript error: {e}")


@app.route("/upload", methods=["POST"])
def upload():
    if "url" in request.form:
        text = extract_text_from_url(request.form["url"])
        return jsonify(
            {"success": True, "message": "🌐 Website added to knowledge base."}
        )

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})

    file = request.files["file"]
    ext = os.path.splitext(file.filename)[1].lower()
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    if ext == ".pdf":
        extract_text_from_pdf(path)
    elif ext==".docx":
        extract_text_from_docx(path)
    elif ext == ".txt":
        extract_text_from_txt(path)
    elif ext in [".mp3", ".m4a", ".wav"]:
        extract_text_from_audio(path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        global last_image_path
        last_image_path = path  # just store the path, do NOT embed
        return jsonify({"success": True, "message": "🖼️ Image stored and ready for visual question answering."})

    else:
        return jsonify({"success": False, "error": "Unsupported file type"})

    return jsonify({"success": True, "message": "✅ Added to knowledge base."})


@app.route("/youtube", methods=["POST"])
def youtube():
    url = request.form["youtube_url"]
    text = get_youtube_text(url)
    return jsonify(
        {"success": True, "message": "🎬 YouTube transcript added.", "text": text}
    )


@app.route("/query", methods=["POST"])
def query():
    global last_image_path

    q = request.json.get("question", "").strip()
    if not q:
        return jsonify({"success": False, "answer": "Question is empty."})

    if vector_store is None:
        return jsonify({"success": False, "answer": "No data added yet!"})

    # Retrieve context from memory
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(q)
    context = "\n".join(d.page_content for d in docs)

    parser = StrOutputParser()

    # Vision model (same model can handle vision + text)
    llm = ChatGroq(model="openai/gpt-oss-120b")

    # ✅ If user uploaded an image, answer using the image
    if last_image_path:
        try:
            with open(last_image_path, "rb") as img:
                response = llm.invoke(
                    {
                        "input": f"Question: {q}\n\nContext:\n{context}",
                        "image": img.read(),
                    }
                )
            return jsonify({"success": True, "answer": response})
        except Exception as e:
            return jsonify(
                {"success": False, "answer": f"Image processing error: {str(e)}"}
            )

    # ✅ Otherwise answer using RAG (text only)
    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant. Use the context below to answer.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer clearly and in simple words:"
        ),
        input_variables=["context", "question"],
    )

    chain = prompt | llm | parser
    answer = chain.invoke({"context": context, "question": q})

    return jsonify({"success": True, "answer": answer})


if __name__ == "__main__":
    app.run(debug=True, port=8000)
