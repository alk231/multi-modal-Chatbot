<img width="1919" height="897" alt="image" src="https://github.com/user-attachments/assets/fd123225-75e8-4a06-b02e-cd54d84cb3a3" />


Multimodal RAG Chatbot

A Retrieval-Augmented Generation (RAG) system that can understand and answer questions across multiple data formats.
The chatbot builds its own knowledge base from PDFs, DOCX, text files, websites, YouTube transcripts, audio, and images, and answers queries using an LLM.

Features

Upload PDF / DOCX / TXT files → text is extracted and stored.

Extract content directly from websites using URL.

Fetch and index YouTube transcripts automatically.

Transcribe audio to text using Whisper.

Ask questions about images (vision-enabled inference).

All processed text is stored in a FAISS vector database, allowing contextual querying.

Architecture

| Component           | Technology                               |
| ------------------- | ---------------------------------------- |
| Backend             | Flask + LangChain                        |
| LLM                 | Groq (`qwen3-32b` / `llava-v1.6-34b`)    |
| Embeddings          | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store        | FAISS (Persistent)                       |
| Audio Transcription | OpenAI Whisper                           |
| Frontend            | React + Tailwind                         |

How It Works

User uploads data or provides a link.

Text is extracted (OCR / parsing / transcription depending on input type).

Text is converted into embeddings and stored in FAISS.

When the user asks a question:

Relevant text chunks are retrieved using similarity search.

The LLM generates a final answer using the retrieved context.

If an image was uploaded, the system switches to vision + language inference.

Project Structure
backend/
│ app.py
│ uploads/
│ faiss_index/     # persistent memory
│
frontend/
│ src/
│   Chatbot.jsx
│

Setup
Backend
 cd backend
 pip install -r requirements.txt
 python app.py

 Runs at http://localhost:8000

Frontend
 cd frontend
 npm install
 npm run dev
Runs at http://localhost:5173

Example Use Flow

Upload a PDF/Docx of medical guidelines.

Paste a YouTube lecture link.

Upload an image like a medical scan.

Ask:

Summarize the treatment recommendations mentioned in the document.


Then ask:

Describe what is shown in the uploaded image.

Future Enhancements

Conversation memory across sessions

Embedding cleanup and deletion UI

PDF viewer inside the frontend
