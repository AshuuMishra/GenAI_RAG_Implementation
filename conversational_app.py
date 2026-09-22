#RAG Conversational with PDF INcluding Chat history
import os 
import uuid

import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma,FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from langchain_core.chat_history import BaseChatMessageHistory , InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser





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
    # model_name='all-MiniLM-L6-v2'
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

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

    documents= []
    #process uploaded pdf
    for file in files:

        with open("uploaded_files.pdf","wb") as f:
            f.write(file.getbuffer())
    

            loader=PyPDFLoader("uploaded_files.pdf")
            docs = loader.load()
            documents.extend(docs)

    #split and create embedding for the documents
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splits=text_splitter.split_documents(documents)
    # vectorstore = Chroma.from_documents(documents=splits ,embedding = embedding)

    st.write("Number of documents:", len(documents))

    for i, doc in enumerate(documents):
        st.write("Page", i, "text length:", len(doc.page_content))

    st.write("First page text:", documents[0].page_content[:500])

    st.write("Number of documents:", len(documents))
    st.write("Number of splits:", len(splits))

    vectorstore = FAISS.from_documents(documents=splits ,embedding = embedding)

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

    # question_rewritter= (
    #     contextualize_q_prompt 
    #     | llm 
    #     | StrOutputParser() 
    #     )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
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

    question_answer_chain=create_stuff_documents_chain(
        llm,
        qa_prompt,
        )

    #Combining the histroy_aware_retrieval as well the question answer chain 
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )


    def get_session_histroy(session : str )-> BaseChatMessageHistory:

        if session not in st.session_state.store:
            st.session_state.store[session] = InMemoryChatMessageHistory()

        return st.session_state.store[session]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_histroy,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
        
    )



    user_input = st.chat_input("Ask question about your pdf :")

    if user_input:

        response = conversational_rag_chain.invoke(
            {"input" : user_input},
            config ={"configurable":{"session_id":session_id}}
        )

        st.write(response['answer'])
        
        # Display chat history
    if session_id in st.session_state.store:

        session_history = st.session_state.store[session_id]

        for message in session_history.messages:

            if message.type == "human":
                st.chat_message("user").write(message.content)

            elif message.type == "ai":
                st.chat_message("assistant").write(message.content)
#         )
#         st.write(st.session_state.store)
#         st.success("assistant:", response["answer"])
#         st.write("chat history", session_history.messages)

else :
    st.warning("please enter the groq api key ")
        