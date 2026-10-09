# Document Chat Bot (RAG) with LangChain, Groq and FAISS

Step-by-step guide for the rag-demo project on Ubuntu, written 9 October 2026

## 1. What you are building

A chat bot that answers questions from your own PDFs, text files and YouTube videos. This technique is called RAG (Retrieval-Augmented Generation). It has two phases.

**Phase A: Ingest (run once, or when your documents change)**

Documents are read, cut into small chunks, converted to numbers (embeddings) and saved in a searchable folder called faiss\_index.

**Phase B: Ask (every question)**

Your question is converted to numbers, the 4 most similar chunks are found, and those chunks plus your question are sent to a Groq AI model, which writes the answer from them.

```
INGEST:  files / videos -> chunks -> embeddings -> faiss_index (saved)
ASK:     question -> embedding -> find 4 closest chunks -> prompt -> Groq model -> answer + sources
```

## 2. The concepts, explained simply

Think of a library: books are cut into index cards, each card gets a meaning code, a card catalog stores the codes, and a librarian finds matching cards.

| Library idea | RAG term | Job |
| --- | --- | --- |
| Index cards | Chunks | Small pieces of text, about 800 characters each |
| Meaning code | Embedding | A list of 384 numbers that represents the meaning of a text |
| Card catalog | Vector store (FAISS) | Stores every chunk with its numbers and finds the closest ones fast |
| Librarian | Retriever | Returns the top k matching chunks for a question |
| Writer | LLM (Groq model) | Writes the final answer using only the chunks it is given |

### 2.1 Chunks

A chunk is a short piece of a document. A 155-page PDF became 501 chunks in this project.

Why not send the whole PDF to the AI? Models have size limits, long inputs are slow and use up free-tier limits, and too much text distracts the model. Four relevant chunks give better answers.

Example: the text 'Lists can be changed after creation. Tuples cannot be changed.' might become Chunk 1 (lists) and Chunk 2 (tuples). A question about tuples finds only Chunk 2.

- chunk\_size=800 means each chunk is up to about 800 characters.
- chunk\_overlap=100 means neighbouring chunks share 100 characters, so an idea is not cut in half at a border.
- Too small loses context. Too big mixes topics and makes matching fuzzy.

### 2.2 Embeddings

Computers match exact words, so a search for 'modify a tuple' would miss 'tuples cannot be changed'. An embedding model turns text into a vector (list of numbers) so that similar meanings get similar numbers.

| Text | Result |
| --- | --- |
| Tuples cannot be changed | close to the next line |
| Tuples are immutable | close to the line above |
| I love pizza | far from both |

Key points:

- The same model must embed both your chunks and your question.
- This project uses all-MiniLM-L6-v2 (about 90 MB, runs on your machine, free).
- The embedding model only makes numbers. It does not write answers. Groq writes the answers.

### 2.3 Vector store and the faiss\_index folder

FAISS is a library that stores vectors and quickly finds the nearest ones. ingest.py saves it to a folder called faiss\_index.

| File inside faiss\_index | Contents |
| --- | --- |
| index.faiss | The vectors, organized for fast search |
| index.pkl | The chunk text and metadata (file name, page number, video link) |

Both files are needed. Saving the folder means startup takes seconds instead of re-embedding everything each time.

Rebuild it (run python ingest.py) when you add, remove or change documents or YouTube URLs, change chunk settings, or change the embedding model. You do not need to rebuild when you change the Groq model, k, or the prompt. It is safe to delete, because it can always be rebuilt from your originals.

Security note: index.pkl is a Python pickle file, which can run code when loaded. That is why load\_local needs allow\_dangerous\_deserialization=True. Only do this for an index you built yourself.

### 2.4 Retriever, prompt and chain

- The retriever wraps the vector store: given a question, return the top k=4 chunks.
- The prompt tells the model to answer only from the supplied context and to say I don't know otherwise. This stops it from inventing answers.
- The chain connects the steps in order: retrieve chunks, fill the prompt, call the model, return plain text.

## 3. Project layout

```
~/rag-demo/
  .venv/                 virtual environment
  docs/                  your PDFs and .txt files
  youtube_urls.txt       one YouTube URL per line
  faiss_index/           created by ingest.py
  ingest.py              builds the index
  rag.py                 terminal chat
  app.py                 web chat (Streamlit)
  assets/agent.png       optional chat icon
```

## 4. Setup, step by step

### Step 1: Create a workspace

```bash
mkdir -p ~/rag-demo && cd ~/rag-demo
python3 -m venv .venv
source .venv/bin/activate
pip install youtube-transcript-api
```

The virtual environment keeps this project's packages separate. Leave it with deactivate. Enter it again with source .venv/bin/activate. Use Python 3.12 or 3.13 if possible, because Python 3.14 is very new and some packages do not have builds for it yet.

### Step 2: Install packages

```bash
pip install langchain langchain-groq langchain-community langchain-huggingface \
  langchain-text-splitters faiss-cpu sentence-transformers pypdf streamlit \
  youtube-transcript-api torchvision
```

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

### Step 5: Add documents

```bash
mkdir -p docs
wget --user-agent='Mozilla/5.0' -O docs/tutorial.pdf '<direct-pdf-link>'
file docs/tutorial.pdf
```

file must say PDF document. If it says HTML document, the download failed (this happened with the bugs.python.org link) and you need a different PDF. Text files can simply be copied into docs/.

### Step 6: Add YouTube videos (optional)

Create youtube\_urls.txt, one URL per line. Lines starting with # are ignored.

```text
# Python tutorials
https://www.youtube.com/watch?v=t2_Q2BRzeEE&list=PLGjplNEQ1it8-0CmoljS5yeV-GlKSUEt0
https://youtu.be/VIDEO_ID_2
```

Only videos with captions work. YouTube often blocks transcript requests from cloud servers (errors like IpBlocked). If that happens, run ingest.py on your laptop and copy the index to the server with scp -r faiss\_index ubuntu@SERVER\_IP:\~/rag-demo/.

## 5. The code

### 5.1 ingest.py (builds the index)

```python
# ingest.py - run once to build the searchable index
import os, re
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi

def video_id_from_url(url):
    m = re.search(r'(?:v=|youtu[.]be/|embed/|shorts/)([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else None

def load_youtube(url, group_seconds=60):
    vid = video_id_from_url(url)
    if not vid:
        print('Skipped (bad URL):', url)
        return []
    try:
        transcript = YouTubeTranscriptApi().fetch(vid, languages=['en'])
    except Exception as e:
        print('Skipped', url, type(e).__name__)
        return []
    link = 'https://www.youtube.com/watch?v=' + vid
    docs, buf, start = [], [], 0
    for s in transcript:
        if not buf:
            start = int(s.start)
        buf.append(' '.join(s.text.split()))
        if s.start - start >= group_seconds:
            docs.append(Document(page_content=' '.join(buf),
                                 metadata={'source': link, 'start': start, 'type': 'youtube'}))
            buf = []
    if buf:
        docs.append(Document(page_content=' '.join(buf),
                             metadata={'source': link, 'start': start, 'type': 'youtube'}))
    return docs

# 1. Load files
documents = []
if os.path.isdir('docs'):
    documents += DirectoryLoader('docs', glob='**/*.pdf', loader_cls=PyPDFLoader).load()
    documents += DirectoryLoader('docs', glob='**/*.txt', loader_cls=TextLoader).load()

# 2. Load YouTube transcripts
if os.path.exists('youtube_urls.txt'):
    for line in open('youtube_urls.txt'):
        url = line.strip()
        if url and not url.startswith('#'):
            print('Fetching transcript:', url)
            documents += load_youtube(url)

print('Loaded', len(documents), 'documents')
if not documents:
    raise SystemExit('Nothing to index. Add files to docs/ or URLs to youtube_urls.txt')

# 3. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print('Created', len(chunks), 'chunks')

# 4. Embed locally and save
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
FAISS.from_documents(chunks, embeddings).save_local('faiss_index')
print('Index saved to faiss_index/')
```

What each part does:

| Part | Explanation |
| --- | --- |
| DirectoryLoader + PyPDFLoader | Reads every PDF in docs/, one document per page, remembering file name and page number |
| load\_youtube | Downloads the video captions and groups them into 60-second documents, remembering the video link and start time |
| RecursiveCharacterTextSplitter | Cuts documents into chunks, preferring paragraph and sentence breaks |
| HuggingFaceEmbeddings | Loads the local embedding model |
| FAISS.from\_documents(...).save\_local | Embeds every chunk and writes the faiss\_index folder |

### 5.2 rag.py (terminal chat)

```python
# rag.py - ask questions in the terminal
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

assert os.getenv('GROQ_API_KEY'), 'Set GROQ_API_KEY first'
MODEL = os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
llm = ChatGroq(model=MODEL, temperature=0)

prompt = ChatPromptTemplate.from_template(
'''Answer the question using only the context below.
If the answer is not in the context, say I don't know.

Context:
{context}

Question: {question}'''
)

def format_docs(docs):
    return '\n\n'.join(d.page_content for d in docs)

def source_label(d):
    m = d.metadata
    if m.get('type') == 'youtube':
        return m['source'] + '&t=' + str(m['start']) + 's'
    name = os.path.basename(str(m.get('source', '?')))
    return name + ' (page ' + str(int(m.get('page', 0)) + 1) + ')'

chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

def ask(q):
    try:
        print()
        print(chain.invoke(q))
        print()
        print('Sources:')
        for s in sorted({source_label(d) for d in retriever.invoke(q)}):
            print('  ', s)
    except Exception as e:
        msg = str(e)
        if 'model_not_found' in msg:
            print('[Error] Model', MODEL, 'not available. Set GROQ_MODEL to a model from your list.')
        elif '429' in msg:
            print('[Error] Groq rate limit hit. Wait a minute and retry.')
        elif '401' in msg or 'invalid_api_key' in msg:
            print('[Error] Invalid GROQ_API_KEY. Export it again.')
        else:
            print('[Error]', msg)

if __name__ == '__main__':
    print('Using model:', MODEL)
    while True:
        q = input('Ask (or quit): ').strip()
        if q.lower() in ('quit', 'exit'):
            break
        if q:
            ask(q)
```

How the chain works, reading left to right:

```
question -> retriever finds 4 chunks -> format_docs joins them into one text
         -> prompt fills {context} and {question}
         -> Groq model writes the answer
         -> StrOutputParser returns plain text
```

An example of what the model actually receives for the question 'Can I change a tuple?':

```text
Answer the question using only the context below.
If the answer is not in the context, say I don't know.

Context:
Tuples are immutable, so once created they cannot be changed.

Question: Can I change a tuple?
```

temperature=0 makes answers consistent and factual instead of creative.

### 5.3 app.py (web chat with history and sources)

```python
# app.py - run with: streamlit run app.py
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

USER_AVATAR = '🧑'
BOT_AVATAR = '🤖'   # or a path such as 'assets/agent.png'

st.set_page_config(page_title='Document Chat', page_icon=BOT_AVATAR)
st.title('Chat with your documents')

if not os.getenv('GROQ_API_KEY'):
    st.error('GROQ_API_KEY is not set. Export it and restart.')
    st.stop()
MODEL = os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')

@st.cache_resource(show_spinner='Loading index and model...')
def load_components():
    emb = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vs = FAISS.load_local('faiss_index', emb, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={'k': 4}), ChatGroq(model=MODEL, temperature=0)

retriever, llm = load_components()

condense_prompt = ChatPromptTemplate.from_template(
'''Given the chat history and a follow-up question, rewrite the follow-up as a
standalone question. If it is already standalone, return it unchanged.
Return only the question.

Chat history:
{history}

Follow-up: {question}'''
)
answer_prompt = ChatPromptTemplate.from_template(
'''Answer the question using only the context below.
If the answer is not in the context, say I don't know.

Context:
{context}

Question: {question}'''
)
condense_chain = condense_prompt | llm | StrOutputParser()
answer_chain = answer_prompt | llm | StrOutputParser()

def source_label(d):
    m = d.metadata
    if m.get('type') == 'youtube':
        return m['source'] + '&t=' + str(m['start']) + 's'
    name = os.path.basename(str(m.get('source', '?')))
    return name + ' (page ' + str(int(m.get('page', 0)) + 1) + ')'

with st.sidebar:
    st.caption('Model: ' + MODEL)
    if st.button('Clear chat'):
        st.session_state.messages = []
        st.rerun()

if 'messages' not in st.session_state:
    st.session_state.messages = []

def show_sources(sources):
    if sources:
        with st.expander('Sources'):
            for s in sources:
                st.write('- ' + s)

for m in st.session_state.messages:
    avatar = BOT_AVATAR if m['role'] == 'assistant' else USER_AVATAR
    with st.chat_message(m['role'], avatar=avatar):
        st.markdown(m['content'])
        if m['role'] == 'assistant':
            show_sources(m.get('sources'))

question = st.chat_input('Ask about your documents')
if question:
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user', avatar=USER_AVATAR):
        st.markdown(question)
    with st.chat_message('assistant', avatar=BOT_AVATAR):
        try:
            with st.spinner('Thinking...'):
                prior = st.session_state.messages[:-1][-6:]
                standalone = question
                if prior:
                    history = '\n'.join(m['role'] + ': ' + m['content'] for m in prior)
                    standalone = condense_chain.invoke(
                        {'history': history, 'question': question}).strip()
                docs = retriever.invoke(standalone)
                context = '\n\n'.join(d.page_content for d in docs)
                answer = answer_chain.invoke({'context': context, 'question': standalone})
                sources = sorted({source_label(d) for d in docs})
        except Exception as e:
            answer, sources = 'Error: ' + str(e), []
        st.markdown(answer)
        show_sources(sources)
    st.session_state.messages.append(
        {'role': 'assistant', 'content': answer, 'sources': sources})
```

What the web app adds on top of rag.py:

| Code | Purpose |
| --- | --- |
| st.cache\_resource | Loads the index and model once, so the app stays fast |
| st.session\_state.messages | Remembers the conversation while the page is open |
| condense\_chain | Rewrites a follow-up such as 'And a tuple?' into a standalone question like 'What is a tuple in Python?' using the recent history, so the search works |
| avatar= in st.chat\_message | Sets the icon next to each message (emoji, image path or URL) |
| show\_sources | Puts the file and page, or the video link at the right second, in a Sources dropdown |

History is kept per browser session, so refreshing the page clears it.

## 6. Running it

### Build the index

```bash
cd ~/rag-demo && source .venv/bin/activate
export HF_HUB_DISABLE_XET=1
python ingest.py
```

Expected output ends with Index saved to faiss\_index/. The first run downloads the 90 MB embedding model.

### Terminal chat

```bash
export GROQ_API_KEY='your-key'
export GROQ_MODEL='openai/gpt-oss-120b'
python rag.py
```

### Web chat

```bash
streamlit run app.py
```

Open http://localhost:8501. On a remote server, create an SSH tunnel from your own computer first, then open the same address locally:

```bash
ssh -L 8501:localhost:8501 ubuntu@SERVER_IP
```

Avoid opening port 8501 to the whole internet, because anyone could then use your Groq key through the app.

### Daily routine after setup

```bash
cd ~/rag-demo
source .venv/bin/activate
export GROQ_API_KEY='your-key'
export GROQ_MODEL='openai/gpt-oss-120b'
streamlit run app.py
```

## 7. Troubleshooting (problems seen in this project)

| Symptom | Cause | Fix |
| --- | --- | --- |
| curl: Binary output can mess up your terminal | A PDF cannot be printed to the screen | Save it with -o or use wget |
| invalid pdf header: b'\<html' | The download was a web page, not a PDF | Check with the file command and download a real PDF |
| CAS Client Error, then OSError: Can't load the model | Network blocked the fast Hugging Face (Xet) download server | export HF\_HUB\_DISABLE\_XET=1, delete the partial cache in \~/.cache/huggingface/hub, run again |
| Unauthenticated requests to the HF Hub warning | No Hugging Face token | Harmless. Set HF\_TOKEN only if you want higher download limits |
| langchain-community sunset DeprecationWarning | The package is being phased out | Harmless for this demo |
| model\_not\_found (404) | Groq removed that model | Use openai/gpt-oss-120b or openai/gpt-oss-20b from your own model list |
| 429 error | Free-tier rate limit | Wait a minute and retry |
| ModuleNotFoundError: torchvision | A dependency imports it | pip install torchvision, or recreate the venv with Python 3.12 or 3.13 |
| YouTube IpBlocked or RequestBlocked | YouTube blocks cloud server addresses | Run ingest.py on your laptop and copy faiss\_index to the server |
| Answer says I don't know | The documents do not cover the question | Ask something the documents contain. This is the intended behaviour |

## 8. Tuning and improving

| Setting | Increase it if | Decrease it if |
| --- | --- | --- |
| chunk\_size | Answers lack context | Results feel unfocused |
| chunk\_overlap | Answers miss information at chunk edges | You want fewer duplicate chunks |
| k | The answer needs information from many places | Extra chunks confuse the model |

Changing chunk\_size or chunk\_overlap requires running ingest.py again. Changing k does not.

To debug bad answers, look at retrieval first. If the wrong chunks come back, the problem is chunking or the embedding model, not the Groq model:

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

emb = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vs = FAISS.load_local('faiss_index', emb, allow_dangerous_deserialization=True)
for doc, score in vs.similarity_search_with_score('Can I modify a tuple?', k=3):
    print(round(float(score), 2), doc.metadata)
    print(doc.page_content[:200])
```

For FAISS, a lower score means a closer match.

## 9. Common mix-ups

| Mix-up | The truth |
| --- | --- |
| The embedding model writes the answer | It only makes numbers. The Groq model writes the answer |
| The AI read my whole PDF | It only sees the 4 chunks sent with each question |
| A bigger k is always better | Unrelated chunks can confuse the model |
| The Groq model must match the embedding model | They are independent. Only the embedding model for ingest and for questions must match |

Groq is free within limits and its model list changes often, so check console.groq.com/docs/models if something stops working.
