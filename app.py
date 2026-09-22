from langchain_core import output_parsers
# import groq
from langchain_core.messages import SystemMessage
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

# langsmith tracking
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING"]="TRUE"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot with GroqAPI"

prompt = ChatPromptTemplate.from_messages([
    ("system" , "you are a helpful assistant.please response to the user queries"),
    ("user","Question:{question}")
])

def generate_response(question, llm , api_key , temperature , max_tokens ):
    groq_api_key = api_key 
    llm = ChatGroq(model=llm)
    output_parsers = StrOutputParser()   #a chat model normally return an AI message object
    chain = prompt | llm | output_parsers
    answer = chain.invoke({"question":question})
    return answer

## title of the app
st.title("Enhanced Q&A Chatbot with GROQ AI")

#sidebar for setting 
st.sidebar.title("Setting")
api_key = st.sidebar.text_input("Enter you Groq AI API Key",type = "password")

#drop down to select varios groq AI models
llm=st.sidebar.selectbox("Select an Groq AI Model",["openai/gpt-oss-120b","openai/gpt-oss-20b","qwen/qwen3.8-27b","groq/compound","groq/compound-mini"])

# adjust response parameter
temperature=st.sidebar.slider("Temperature",max_value=1.0, min_value=0.0)
max_tokens=st.sidebar.slider("Max Tokens",max_value=50, min_value=100)

#main interface for users
st.write("Go Ahead and Start Exploring your Ideas")
user_input = st.text_input("You :")

if user_input:
    response = generate_response(user_input ,llm  ,api_key, temperature , max_tokens)
    st.write(response)
else:
    st.write("Kuch toh likh loudu")