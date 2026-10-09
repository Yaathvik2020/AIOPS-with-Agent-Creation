# app.py
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="KSYS Chatbot", page_icon="🤖")
st.title("🤖 KSYS AI Agent")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Run `export GROQ_API_KEY=...` and restart.")
    st.stop()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


@st.cache_resource(show_spinner="Loading index and model...")
def load_components():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True  # your own index
    )
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    llm = ChatGroq(model=MODEL, temperature=0)
    return retriever, llm


retriever, llm = load_components()

condense_prompt = ChatPromptTemplate.from_template(
    """Given the chat history and a follow-up question, rewrite the follow-up as a
standalone question. If it is already standalone, return it unchanged.
Return only the question.

Chat history:
{history}

Follow-up: {question}"""
)

answer_prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}"""
)

condense_chain = condense_prompt | llm | StrOutputParser()
answer_chain = answer_prompt | llm | StrOutputParser()

# ---------- sidebar ----------
with st.sidebar:
    st.caption(f"Model: `{MODEL}`")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- chat state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources):
    if sources:
        with st.expander("Sources"):
            for s in sources:
                st.write(f"- {s}")


# replay history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_sources(m.get("sources"))

# ---------- new question ----------
if question := st.chat_input("Ask about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                # use the last few turns to resolve follow-ups
                prior = st.session_state.messages[:-1][-6:]
                standalone = question
                if prior:
                    history = "\n".join(f"{m['role']}: {m['content']}" for m in prior)
                    standalone = condense_chain.invoke(
                        {"history": history, "question": question}
                    ).strip()

                docs = retriever.invoke(standalone)
                context = "\n\n".join(d.page_content for d in docs)
                answer = answer_chain.invoke({"context": context, "question": standalone})

                sources = sorted(
                    {
                        f"{os.path.basename(str(d.metadata.get('source', '?')))} "
                        f"(page {int(d.metadata.get('page', 0)) + 1})"
                        for d in docs
                    }
                )
        except Exception as e:
            answer, sources = f"Error: {e}", []

        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
