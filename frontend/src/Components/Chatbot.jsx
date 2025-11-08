import { useState, useRef, useEffect } from "react";
import {
  FaPaperPlane,
  FaFileUpload,
  FaMicrophone,
  FaLink,
  FaYoutube,
  FaImage,
} from "react-icons/fa";

export default function Chatbot() {
  const API_BASE = "http://localhost:8000";

  const [messages, setMessages] = useState([
    {
      sender: "bot",
      text: "Hello! Upload files, images, audio, webpage URLs or YouTube links. Then ask questions from them.",
    },
  ]);
  const [input, setInput] = useState("");
  const [urlInput, setUrlInput] = useState("");
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const chatEndRef = useRef(null);

  const fileInputRef = useRef(null);
  const imageInputRef = useRef(null);
  const audioInputRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const askQuery = async () => {
    if (!input.trim()) return;

    setMessages((prev) => [...prev, { sender: "user", text: input }]);
    const question = input;
    setInput("");

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: data.answer || "⚠️ No answer available." },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: `❌ Error: ${error.message}` },
      ]);
    }
  };

  const handleFileUpload = async (file, label) => {
    setMessages((prev) => [
      ...prev,
      { sender: "user", text: `Uploading ${label}: ${file.name}...` },
    ]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: data.success ? `✅ ${data.message}` : `❌ ${data.error}`,
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: `❌ Upload error: ${error.message}` },
      ]);
    }
  };

  const handleUrlSubmit = async () => {
    if (!urlInput.trim()) return;

    setMessages((prev) => [
      ...prev,
      { sender: "user", text: `🌐 Adding webpage: ${urlInput}` },
    ]);

    const formData = new FormData();
    formData.append("url", urlInput);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: data.success ? `✅ Webpage added.` : `❌ ${data.error}`,
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: `❌ Error: ${error.message}` },
      ]);
    }

    setUrlInput("");
  };

  const handleYouTubeSubmit = async () => {
    if (!youtubeUrl.trim()) return;

    setMessages((prev) => [
      ...prev,
      { sender: "user", text: `🎥 Processing YouTube: ${youtubeUrl}` },
    ]);

    const formData = new FormData();
    formData.append("youtube_url", youtubeUrl);

    try {
      const res = await fetch(`${API_BASE}/youtube`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: data.success ? `✅ Transcript added.` : `❌ ${data.error}`,
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: `❌ Error: ${error.message}` },
      ]);
    }

    setYoutubeUrl("");
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <div className="bg-white p-4 shadow text-center text-lg font-semibold">
        Multimodal RAG Chatbot
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${
              msg.sender === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`px-4 py-2 whitespace-pre-line rounded-2xl max-w-lg ${
                msg.sender === "user"
                  ? "bg-blue-500 text-white rounded-br-none"
                  : "bg-white text-gray-800 shadow rounded-bl-none"
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      <div className="bg-white p-3 flex flex-col gap-3 shadow-md">
        {/* URL */}
        <div className="flex items-center gap-2">
          <FaLink />
          <input
            type="text"
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
            placeholder="Enter website URL..."
            className="flex-1 border p-2 rounded"
            onKeyDown={(e) => e.key === "Enter" && handleUrlSubmit()}
          />
          <button
            className="bg-green-500 text-white px-3 py-2 rounded"
            onClick={handleUrlSubmit}
          >
            Add
          </button>
        </div>

        {/* YouTube */}
        <div className="flex items-center gap-2">
          <FaYoutube className="text-red-500" />
          <input
            type="text"
            value={youtubeUrl}
            onChange={(e) => setYoutubeUrl(e.target.value)}
            placeholder="Enter YouTube link..."
            className="flex-1 border p-2 rounded"
            onKeyDown={(e) => e.key === "Enter" && handleYouTubeSubmit()}
          />
          <button
            className="bg-red-500 text-white px-3 py-2 rounded"
            onClick={handleYouTubeSubmit}
          >
            Fetch
          </button>
        </div>

        {/* Upload + Ask */}
        <div className="flex items-center gap-3">
          {/* PDF/TXT */}
          <input
            type="file"
            ref={fileInputRef}
            style={{ display: "none" }}
            onChange={(e) => handleFileUpload(e.target.files[0], "file")}
          />
          <button
            onClick={() => fileInputRef.current.click()}
            className="p-2 hover:bg-gray-200 rounded"
          >
            <FaFileUpload />
          </button>

          {/* Image */}
          <input
            type="file"
            accept="image/*"
            ref={imageInputRef}
            style={{ display: "none" }}
            onChange={(e) => handleFileUpload(e.target.files[0], "image")}
          />
          <button
            onClick={() => imageInputRef.current.click()}
            className="p-2 hover:bg-gray-200 rounded"
          >
            <FaImage />
          </button>

          {/* Audio */}
          <input
            type="file"
            accept="audio/*"
            ref={audioInputRef}
            style={{ display: "none" }}
            onChange={(e) => handleFileUpload(e.target.files[0], "audio")}
          />
          <button
            onClick={() => audioInputRef.current.click()}
            className="p-2 hover:bg-gray-200 rounded"
          >
            <FaMicrophone />
          </button>

          {/* Ask */}
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask anything..."
            className="flex-1 border p-2 rounded"
            onKeyDown={(e) => e.key === "Enter" && askQuery()}
          />
          <button
            onClick={askQuery}
            className="bg-blue-500 text-white p-2 rounded"
          >
            <FaPaperPlane />
          </button>
        </div>
      </div>
    </div>
  );
}
