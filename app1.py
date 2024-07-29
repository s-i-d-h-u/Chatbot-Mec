import streamlit as st
from langchain_community.document_loaders import TextLoader
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']


static_context = """
The context of this chatbot is to provide information about the Model Engineering College (MEC) website 
direct user to official website if the query is out of scope
"""

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = TextLoader("./intents2.txt")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


st.title("chatbot-mec")
llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="mixtral-8x7b-32768")

prompt = ChatPromptTemplate.from_template(
    f"""
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the input query
    {{context}}
    {static_context}
    Questions: {{input}}
    """,
    input_vars=["context"]
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)
prompt = st.text_input("input your prompt")

if prompt:
    start = time.process_time()
    response = retriever_chain.invoke({"input": prompt, "context": static_context})
    print("response time :", time.process_time() - start)
    st.write(response['answer'])
