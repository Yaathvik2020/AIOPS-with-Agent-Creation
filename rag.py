# rag.py
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

assert os.getenv("GROQ_API_KEY"), "Set GROQ_API_KEY first"

# Change the model without editing code: export GROQ_MODEL="model-name"
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Load the saved index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True  # safe: it's your own index
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Groq LLM
llm = ChatGroq(model=MODEL, temperature=0)

prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}"""
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask(q: str):
    try:
        print("\n" + chain.invoke(q))
        # Show which pages the answer came from
        docs = retriever.invoke(q)
        sources = sorted({(d.metadata.get("source", "?"), d.metadata.get("page", "?")) for d in docs})
        print("\nSources:", ", ".join(f"{s} (page {p})" for s, p in sources))
    except Exception as e:
        msg = str(e)
        if "model_not_found" in msg or "404" in msg:
            print(f"\n[Error] Model '{MODEL}' not available. List models with the curl command, "
                  f"then run: export GROQ_MODEL=\"<model-id>\"")
        elif "429" in msg or "rate" in msg.lower():
            print("\n[Error] Groq rate limit hit. Wait a minute and try again.")
        elif "401" in msg or "invalid_api_key" in msg:
            print("\n[Error] Invalid GROQ_API_KEY. Check the key and export it again.")
        else:
            print(f"\n[Error] {msg}")

if __name__ == "__main__":
    print(f"Using model: {MODEL}")
    while True:
        q = input("\nAsk (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if q:
            ask(q)
