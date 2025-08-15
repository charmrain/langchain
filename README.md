# 📚 Custom Chatbot with Document & Website Ingestion

A LangChain + Streamlit powered chatbot capable of answering questions from **your own documents and websites**.  
This project started as a simple PDF Q&A tool and has grown into a multi-source, multi-file ingestion system.

---

## 🚀 Features

### Version 1 – Single PDF Ingestion
- Upload **one PDF** and ask questions directly about its content.
- Simple interface powered by [Streamlit](https://streamlit.io/) and [LangChain](https://www.langchain.com/).
- [View code → `app.py`](https://github.com/charmrain/langchain/blob/main/streamlit/app.py)

### Version 2 – Multiple PDF Ingestion
- Support for **multiple PDFs** in a single session.
- Automatically indexes all uploaded files for faster retrieval.
- [View code → `chatai_lite.py`](https://github.com/charmrain/langchain/blob/main/streamlit/chatai_lite.py)

### Version 3 – PDF + Website Ingestion
- Upload PDFs **and** scrape content from a given website URL.
- Combine knowledge from multiple sources for richer answers.
- [View code → `tri_app.py`](https://github.com/charmrain/langchain/blob/main/streamlit/tri_app.py)

---

## 🛠️ Tech Stack

- **Python 3.10+**
- [LangChain](https://www.langchain.com/) – LLM orchestration & document processing
- [Streamlit](https://streamlit.io/) – Interactive web UI
- [PyPDF2](https://pypi.org/project/PyPDF2/) – PDF text extraction
- [BeautifulSoup4](https://pypi.org/project/beautifulsoup4/) – Website content scraping
- [OpenAI API](https://platform.openai.com/) – Language model integration

---

## 📥 Installation

```bash
# Clone the repository
git clone https://github.com/charmrain/pdf-web-qa-bot.git
cd langchain/streamlit

## Usage
# Version 1: Single PDF
streamlit run app.py

# Version 2: Multiple PDFs
streamlit run chatai_lite.py

# Version 3: PDFs + Website
streamlit run tri_app.py
