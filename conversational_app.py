#RAG Conversational with PDF INcluding Chat history
from langchain_core.chat_history import BaseChatMessageHistory
import os 
import uuid

import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from lanchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,AIMessage

# environment variable
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

groq_api_key = os.getenv("GROQ_API_KEY")

# langsmith tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING"]="TRUE"
os.environ["LANGCHAIN_PROJECT"]="RAG Document Q&A with GroqAPI"


#embedding models
embedding = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2'
)

#STREAMLIT APP SETUP
st.title("Conversational RAG with PDF upload and chat history")

st.write("Upload a PDF and chat with its content")

#input thr groq api key
api_key = st.text_input("Enter your groq api Key", type ="password")

files = st.file_uploader("choose a pdf file", type="pdf",accept_multiple_files=True)

#check if groq api key is provided
if api_key:
    llm = ChatGroq(groq_api_key=api_key,model="openai/gpt-oss-20b")

    #chat interfrace
    session_id = st.text_input("session ID ", value = "default_session")

    #Chat history store for AI and Human message in dict 
    if "store" not in st.session_state:
        st.session_state.store={}  


    #process uploaded pdf
    for file in files:

        documents= []

        with open("uploaded_files.pdf","wb") as f:
            f.write(file.getbuffer())
    

            loader=PyPDFLoader("uploaded_files.pdf")
            docs = loader.load()
            documents.extend(docs)

    #split and create embedding for the documents
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splits=text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits ,embedding = embedding)
    retriever = vectorstore.as_retriever()


    contextualize_ques_system_prompt = (
        """ Given a chat history and the latest user question which might 
        reference context in that chat history , 
        formulate a standaline question which can be understood without that
        chat history.

        Do not answer the question.

        if the question is already standalone 
        return it unchanged.""" 
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([

        ("system",contextualize_ques_system_prompt),
        
        (
        "human",""" conversation Chat history  : {chat_history}
                    current question           : {input}   
                """
        )

    ])
    
    #question Rewritter for reframming the contexual aware question 
    # Understand/rewrite the question.

    question_rewritter= (
        contextualize_q_prompt 
        | llm 
        | StrOutputParser() 
        )

    #asnwer question

    system_prompt = (

        """
        You're an assistant for question answering.
        use the following retrieved context to answer the question.
        if you dont know the answer, say that you dont know.

        keep the answer concise.

        Retrieved context : {context}  
        """
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",    system_prompt),
            
            ("human", """ Chat history     : {chat_history}
                          Current question : {input}
            """)
        
        ]
    )

    question_answer_chain=(
        qa_prompt
        | llm
        | StrOutputParser()
        )

    #Combining the question rewritter as well the question answer chain 
    Rag_chain= (question_rewritter | question_answer_chain)

    def get_session_history(session:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain=RunnableWithMessageHistory(

        rag_chain,get_session_history,
        input_messages_key="input",
        history_messages_key="chathistory",
        output_messages_key="answer"
    )

    user_input = st.text_input("Your question:")
    if user_input:
        session_history=get_session_history(session_id)
        response = conversational_rag_chain.invoke (
            {"input":user_input},
            config = {
                "configurable":{"session_id":session_id}
            },

        )
        st.write(st.session_state.store)
        st.success("assistant:", response["answer"])
        st.write("chat history", session_history.messages)

else :
    st.warning("please enter the groq api key ")
        