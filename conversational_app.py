#RAG Conversational with PDF INcluding Chat history
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

#check if groq api key is provided
if api_key:
    llm = ChatGroq(groq_api_key=api_key,model="openai/gpt-oss-20b")

    #chat interfrace
    session_id = st.text_input("session ID ", value = "default_session")

    #Chat history store for AI and Human message in dict 
    if "store" not in st.session_state:
        st.session_state.store={}  

    uploaded_files = st.file_uploader("choose a pdf file", type="pdf",accept_multiple_files=True)

    #process upladed pdf 
    if uploaded_files:
        documents =[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        #split and create embedding for the documents
        text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        splits=text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits ,embedding = embedding)
        retriever = vectorstore.as_retriever()


    contextualie_q_system_prompt = (
        """ given a chat history and the latest user question 
        which might reference context in the chat history 
        formula to standalone application question which can be understood 
        without chat history, do not answer the question, just reformulate it if needed and otherwise return this."""

    )

    contextualie_q_prompt = ChatPromptTemplate.from_messages([
        ("system",contextualie_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}"),

    ])

    history_aware_retriever = create_history_aware_retriever(llm,retriever, contextualie_q_system_prompt)

    #asnwer question

    system_prompt = (

        """
        you are an assistant for the question answer task.

        Use the following pieces of retrieve context to answer the question.

        If you don't know the answer, say that you don't know.

        Use three sentence maximum and keep the answer concise.
        "\n\n"
        "{context}"
        """
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),
        
        ]
    )

    question_answer_chain=create_stutt_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

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
        