#RAG Document Q&A with Groq API 
import os 
import streamlit as st

from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")

# langsmith tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING"]="TRUE"
os.environ["LANGCHAIN_PROJECT"]="RAG Document Q&A with GroqAPI"


llm = ChatGroq(api_key=groq_api_key,
    model="openai/gpt-oss-20b"
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only 
    please provide the most accurate response based on the question
    <context>
    {context}
    <context>

    Quesstion : {input}
    """

)

def create_vectore_embedding():
    if "vector" not in st.session_state:
        st.session_state.embedding = HuggingFaceEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader(r"C:\Users\CloudJournee\Desktop\python\python\langchain_update\GenAI_RAG_Implementation\attention.pdf")  #data ingestion step
        st.session_state.docs = st.session_state.loader.load()  #document loader
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vector=FAISS.from_documents(st.session_state.final_documents,st.session_state.embedding)


user_prompt = st.text_input("Enter your query from the document")

if st.button("Document Embedding"):
    create_vectore_embedding()
    st.write("You;re vector Database is ready....!")

import time 

if user_prompt:
    
    retriever = st.session_state.vector.as_retriever()

    retrival = (
        {
            "context" :retriever,
            "input" : RunnablePassthrough()
        
        }
        | prompt | llm | StrOutputParser
    )

    start = time.process_time()
    response = retrival.invoke({"input" :user_prompt})
    print(f"Response time :{time.process_time() - start}")

    st.write(response["answer"])
