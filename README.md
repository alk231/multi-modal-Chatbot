<img width="1906" height="894" alt="image" src="https://github.com/user-attachments/assets/dc247732-e80d-423a-880f-ec97c8338764" />

Multimodal RAG Chatbot
This project is a Multimodal Retrieval-Augmented Generation (RAG) chatbot that can understand and answer questions from various types of inputs. You can upload documents, websites, YouTube videos, audio files, and even images. The system extracts text, stores it in a vector database, and uses a Large Language Model (LLM) to answer questions based on the stored knowledge.

Key Features
FeatureDescriptionPDF / DOCX / TXT SupportExtracts text from documents and adds it to the knowledge base.Audio TranscriptionConverts audio to text using Whisper and stores it for querying.Website Content ExtractionFetches and processes content directly from a URL.YouTube Transcript IntegrationRetrieves subtitles from YouTube videos and adds them to the vector store.Image Question AnsweringYou can upload an image and ask questions about it.Persistent Vector MemoryAll uploaded data is stored in FAISS and remains available across sessions.

Tech Stack
LayerTools UsedBackendPython, Flask, LangChainLLMGroq API (qwen/qwen3-32b / llava-v1.6-34b)Embeddingssentence-transformers/all-MiniLM-L6-v2Vector StoreFAISS (persistent storage)Audio ProcessingOpenAI WhisperFrontendReact + Tailwind CSS

How It Works


User uploads data → Documents, URL, YouTube link, audio, or image.


Text is extracted and embedded into vector representation.


Embeddings are stored in FAISS vector database for retrieval.


When the user asks a question:


Relevant text chunks are retrieved from the vector store.


The LLM generates a final answer using both the retrieved context and the question.




If an image was uploaded, the model performs Vision + Text reasoning.



Project Folder Structure
project/
│── backend/
│   ├── app.py                # Flask backend with RAG + vision support
│   ├── uploads/              # Uploaded files stored here
│   ├── faiss_index/          # Persistent vector database
│
└── frontend/
    ├── src/
    │   └── Chatbot.jsx       # React UI for chatting and uploading
    └── package.json


Run the Backend
cd backend
pip install -r requirements.txt
python app.py

Backend runs at:
http://localhost:8000


Run the Frontend
cd frontend
npm install
npm run dev

Frontend runs at:
http://localhost:5173


Example Usage


Upload a PDF or DOCX file.


Upload a YouTube link to add its transcript.


Speak or upload audio; it will be transcribed.


Upload an image, then ask:
What is happening in this picture?



Ask any question based on stored knowledge:
Summarize the document I uploaded.




Future Improvements


Multi-user session-based memory.


UI improvements for document preview.


Support for more embedding models and vector stores.


Fine-grained memory deletion & cleanup.



Credits
Developed by Susobhan Akhuli
Using Groq LLM, LangChain, Whisper, FAISS, and React.

If you want, I can now:
✅ Generate a demo video script
✅ Create screenshots section
✅ Write a LinkedIn post to showcase this project
