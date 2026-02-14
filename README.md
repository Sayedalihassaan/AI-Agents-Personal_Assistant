# AI-Agents Personal Assistant

AI-Agents Personal Assistant is a modern, extensible, and privacy-focused AI-powered assistant that integrates with your email, calendar, web search, and vector database to help you manage your digital life. Built with Python, Streamlit, FastAPI, and LangChain, it offers a modular backend and a user-friendly web interface.

---

## 🚀 Features

- **Conversational AI**: Natural language chat interface powered by LLMs (OpenAI, Google, etc.)
- **Gmail Integration**: Read, send, and manage your Gmail messages securely
- **Google Calendar Integration**: Create, update, and search calendar events
- **Web Search**: Get up-to-date information using DuckDuckGo
- **Vector Database**: Store and retrieve documents using Pinecone
- **Chat History**: Persistent, secure chat history with database support
- **Date & Time Tools**: Query current date/time in any timezone
- **Extensible Tools**: Easily add new tools and integrations
- **Modern UI**: Streamlit-based web interface for easy interaction

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, LangChain, Pinecone, MySQL/PostgreSQL
- **Frontend**: Streamlit
- **Integrations**: Gmail, Google Calendar, DuckDuckGo, Pinecone
- **Authentication**: API tokens, OAuth2 (Google)

---

## 📦 Installation

1. **Clone the repository:**
	```bash
	git clone https://github.com/sayedali/AI-Agents-Personal_Assistant.git
	cd AI-Agents-Personal_Assistant
	```

2. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```

3. **Set up environment variables:**
	- Copy `.env.example` to `.env` and fill in your API keys and credentials.
	- Place your Google credentials in `credentials.json`.

4. **Run the Streamlit app:**
	```bash
	streamlit run streamlit_app.py
	```

5. **(Optional) Run the FastAPI backend:**
	```bash
	uvicorn main:app --reload
	```

---

## ⚙️ Configuration

- **Google API**: Place your OAuth credentials in `credentials.json` and configure token paths in `.env`.
- **Pinecone**: Set your Pinecone API key and index name in `.env`.
- **Database**: Configure your database URL in `.env` for chat history persistence.

---

## 🧩 Project Structure

```
├── main.py                # FastAPI backend
├── streamlit_app.py       # Streamlit frontend
├── backend/               # Core backend logic
│   ├── config.py
│   ├── database/
│   ├── services/
│   │   ├── agent_service/
│   │   ├── tools/
│   │   └── ...
│   └── ...
├── requirements.txt
├── credentials.json
├── README.md
└── ...
```

---

## 📝 Usage

1. Open the Streamlit app in your browser (usually at `http://localhost:8501`).
2. Start chatting with your AI assistant!
3. Use commands like:
	- "Send an email to John about the meeting."
	- "What events do I have tomorrow?"
	- "Search the web for the latest AI news."
	- "What time is it in Tokyo?"

---

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, features, or improvements.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [Streamlit](https://streamlit.io/)
- [Pinecone](https://www.pinecone.io/)
- [DuckDuckGo Search](https://duckduckgo.com/)
- [Google Cloud](https://cloud.google.com/)