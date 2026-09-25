from requests.utils import get_netrc_auth
import validators,streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain_groq import ChatGroq

import os 
from dotenv import load_dotenv
load_dotenv()

groq_api=os.getenv('GROQ_API_KEY')

print("GROQ KEY FOUND:", groq_api is not None)

llm=ChatGroq(
    model="openai/gpt-oss-20b",

)

#Streamlit APP
st.set_page_config(page_title="Langchain: Summarise Text FRom YT or website")
st.title("Langchain : Summarize Text from YT or website")
st.subheader("Summarise URL")


#get the url (YT or website) to be summarised
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key", value="", type = "password")

generic_url =st.text_input("URL",label_visibility="collapsed")

prompt_template = """
Provide me the summary in 300 words
content : {text}
"""

prompt =PromptTemplate(template=prompt_template,input_variables=['text'],verbose=True)

if st.button("Summarise the content from YT or website "):
    #valudate the all the input 
    if not groq_api_key.strip() or not generic_url.strip():
        st.write("Enter the required Input")
        st.error("Please provide the information")

    elif not validators.url(generic_url):
        st.error("Gawar log ! , Url bhi copy paste karne nhi aata hai...")

    else :
        try :
            with st.spinner("waiting..."):
                #loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)

                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                    headers= "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
                docs = loader.load()

                # Cain for summarisation
                chain= load_summarize_chain(
                    prompt=prompt,
                    llm=llm, 
                    chain_type="stuff"
                )
                summarizer=chain.invoke(docs)
                st.success(summarizer)
        
        except Exception as e:
            st.exception(e)

