# Document Chat Bot (RAG) with LangChain, Groq and FAISS

Step-by-step guide for the rag-demo project on Ubuntu, written 9 October 2026

## 1. What you are building

A chat bot that answers questions from your own PDFs, text files and YouTube videos. This technique is called RAG (Retrieval-Augmented Generation). It has two phases.

**Phase A: Ingest (run once, or when your documents change)**

Documents are read, cut into small chunks, converted to numbers (embeddings) and saved in a searchable folder called faiss\_index.

**Phase B: Ask (every question)**

Your question is converted to numbers, the 4 most similar chunks are found, and those chunks plus your question are sent to a Groq AI model, which writes the answer from them.

```
### Step 1: Create a workspace

```bash
mkdir -p ~/rag-demo && cd ~/rag-demo
python3 -m venv .venv
source .venv/bin/activate
pip install youtube-transcript-api
```
### Step 2: Install packages

```bash
pip install langchain langchain-groq langchain-community langchain-huggingface \
  langchain-text-splitters faiss-cpu sentence-transformers pypdf streamlit \
  youtube-transcript-api torchvision \
  youtube-transcript-api
```
INGEST:  files / videos -> chunks -> embeddings -> faiss_index (saved)
ASK:     question -> embedding -> find 4 closest chunks -> prompt -> Groq model -> answer + sources
### Step 3: Get a free Groq API key

Sign up at console.groq.com, open API Keys, create a key, then set it in your terminal:

```bash
export GROQ_API_KEY='your-key'
```

This variable disappears when you close the terminal, so set it again in each new session.

### Step 4: Pick a working Groq model

Groq changes its model list often. List what your key can use:

```bash
curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
```

The Llama chat models returned model\_not\_found, so use a GPT-OSS model from your list:

```bash
export GROQ_MODEL='openai/gpt-oss-120b'
```

openai/gpt-oss-20b is a faster alternative. Skip the whisper, orpheus, prompt-guard and safeguard models, which are for speech or safety, not chat.

## ingest.py -->  configure the PDF/ You tube text.  need to rerun every each add/update of PDF, text file
##python ingest.py
```bash
python ingest.py ' youtube argut'
#run multiple  yoube
export YOUTUBE_URLS='youtube-1, youtube-2'
python ingest.py
'''
### Terminal chat

```bash
export GROQ_API_KEY='your-key'
export GROQ_MODEL='openai/gpt-oss-120b'
python rag.py
```
 ### Web chat
```bash
 pip install streamlit
```
### Run web serv

```bash
export GROQ_API_KEY="your-key"
export GROQ_MODEL="openai/gpt-oss-120b"
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
