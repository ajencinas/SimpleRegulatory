import streamlit as st

# MUST be first
st.set_page_config(page_title="💬 RAG Chat", layout="wide")

import time
import os
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from RAGsearch import load_vector_store
from urllib.parse import quote
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory import ConversationBufferMemory

# 🔁 Streaming callback
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.tokens = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens += token
        self.container.markdown(self.tokens + "▌")

    def final_answer(self):
        self.container.markdown(self.tokens)
        return self.tokens

# 🧠 Load vector store once
@st.cache_resource(show_spinner="Loading vector store...")
def get_vector_store():
    return load_vector_store()

vector_store = get_vector_store()

# 🧠 Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# UI
st.title("💬 RAG Chat Assistant")

with st.chat_message("system"):
    st.markdown("Ask questions about your documents and get streaming answers below:")

user_input = st.chat_input("Ask a question...")


if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        callback = StreamlitCallbackHandler(response_placeholder)

        try:
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            llm = ChatOpenAI(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.1,
                model="gpt-4",
                streaming=True,
                callbacks=[callback]
            )

            # 👇 Build (user, assistant) history from session
            history_pairs = []
            for i in range(0, len(st.session_state.chat_history) - 1, 2):
                user_msg = st.session_state.chat_history[i].get("content", "")
                ai_msg = st.session_state.chat_history[i + 1].get("content", "")
                history_pairs.append((user_msg, ai_msg))

            # ✅ Use ConversationalRetrievalChain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
            )

            start = time.time()
            result = qa_chain({"question": user_input, "chat_history": history_pairs})
            duration = time.time() - start
            final_answer = callback.final_answer()

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": final_answer,
                "duration": f"{duration:.2f}"
            })

            st.caption(f"⏱️ Answer generated in {duration:.2f} seconds")

            sources = result.get("source_documents", [])
            if sources:
                with st.expander("📂 Source Excerpts"):
                    for doc in sources:
                        meta = doc.metadata
                        file_name = meta.get("file_name", "Unknown file")
                        page_number = meta.get("page_number", "?")
                        source_text = doc.page_content.strip()
                        safe_file = quote(file_name)

                        file_url = f"http://localhost:8000/{safe_file}#page={page_number}"
                        st.markdown(
                            f"[📄 **{file_name}** (Page {page_number}) — Click to open PDF ↗️]({file_url})",
                            unsafe_allow_html=True
                        )
                        st.code(source_text, language="markdown")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# Replay history
for msg in st.session_state.chat_history[:-2]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
