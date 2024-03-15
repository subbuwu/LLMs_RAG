import os
import streamlit as st
import torch

from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,load_index_from_storage,ServiceContext,StorageContext

from llama_index.core.indices.vector_store.retrievers.retriever import VectorIndexRetriever

from llama_index.core.response.pprint_utils import pprint_response

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM

from llama_index.embeddings.huggingface import HuggingFaceEmbedding


storage_path = "./vectorstore"
documents_path = "/Users/subbu/Desktop/practice_llms_rags/llama_index_rag/documents"

hf_token = "YOUR_HUGGINGFACE_TOKEN"

llm = HuggingFaceLLM(
    model_name="HuggingFaceH4/zephyr-7b-alpha",
    tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
    query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
    context_window=3900,
    max_new_tokens=256,
    # tokenizer_kwargs={},
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    device_map="auto",
)

Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

@st.cache_resource(show_spinner=False)

def initialize():
    if not os.path.exists(storage_path):
        documents = SimpleDirectoryReader(documents_path).load_data()
        index = VectorStoreIndex.from_documents(documents=documents)
        index.storage_context.persists(persist_dir=storage_path)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        index = load_index_from_storage(storage_context=storage_context)
    return index

index = initialize()

st.title("Ask the document")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        { "role" : "assistant" , "content" : "Ask me a question!" }
    ]

chat_engine = index.as_chat_engine(chat_mode="condense_question",verbose=True)

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({
        "role" : "user",
        "content" : prompt
    })

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant" :
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            pprint_response(response,show_source=True)
            message = {
                "role" : "assistant",
                "content" : response.response
            }
            st.session_state.messages.append(message)

