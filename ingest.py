# ingest.py - builds the searchable index from files and YouTube videos
import os, re, sys
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
    """Fetch captions and group them into ~60-second documents."""
    vid = video_id_from_url(url)
    if not vid:
        print('Skipped (bad URL):', url)
        return []
    try:
        transcript = YouTubeTranscriptApi().fetch(vid, languages=['en'])
    except Exception as e:
        print('Skipped', url, '-', type(e).__name__)
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


def collect_youtube_urls():
    """Gather URLs from: command line, YOUTUBE_URLS env var, youtube_urls.txt"""
    urls = []

    # 1. command line:  python ingest.py URL1 URL2
    urls += sys.argv[1:]

    # 2. environment variable:  export YOUTUBE_URLS='URL1,URL2'
    for u in os.getenv('YOUTUBE_URLS', '').split(','):
        urls.append(u)

    # 3. file: one URL per line, lines starting with # are ignored
    if os.path.exists('youtube_urls.txt'):
        for line in open('youtube_urls.txt'):
            urls.append(line)

    # clean up, drop empty lines and comments, remove duplicates (keep order)
    clean, seen = [], set()
    for u in urls:
        u = u.strip()
        if u and not u.startswith('#') and u not in seen:
            seen.add(u)
            clean.append(u)
    return clean


# 1. Load PDFs and text files
documents = []
if os.path.isdir('docs'):
    documents += DirectoryLoader('docs', glob='**/*.pdf', loader_cls=PyPDFLoader).load()
    documents += DirectoryLoader('docs', glob='**/*.txt', loader_cls=TextLoader).load()

# 2. Load YouTube transcripts
for url in collect_youtube_urls():
    print('Fetching transcript:', url)
    documents += load_youtube(url)

print('Loaded', len(documents), 'documents')
if not documents:
    raise SystemExit('Nothing to index. Add files to docs/ or give YouTube URLs.')

# 3. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print('Created', len(chunks), 'chunks')

# 4. Embed locally and save the index
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
FAISS.from_documents(chunks, embeddings).save_local('faiss_index')
print('Index saved to faiss_index/')
