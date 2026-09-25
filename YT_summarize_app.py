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
Provide a clear, visually structured summary of the following content in approximately 300 words.

Follow this structure:

1. **Title**

   * Create a short, engaging title that captures the main topic.\n

2. **Overview**

   * Give a concise 2–3 sentence introduction explaining what the content is about and why it is important.\n

3. **Key Points**

   * Present the most important ideas as clear bullet points.
   * Focus on concepts, facts, arguments, processes, and important observations.\n

4. **Key Concepts / Knowledge**

   * Highlight important terms, concepts, technologies, people, or ideas.
   * Briefly explain each one in simple language where necessary.\n

5. **Important Takeaways**

   * Provide 3–5 practical or memorable takeaways from the content.\n

6. **Conclusion**

   * End with a short conclusion that connects the main ideas together.\n

Formatting requirements:

* Keep the summary approximately 300 words.
* Use clear headings and bullet points.
* Use **bold text** for important terms and concepts.
* Keep the language concise, professional, and easy to understand.
* Do not introduce information that is not present in the provided content.
* Avoid unnecessary repetition.
* Preserve the original meaning of the content.

Content:
{text}

"""

prompt =PromptTemplate(template=prompt_template,input_variables=['text'],verbose=True)

if st.button("Summarise the content from YT or website "):
    #valudate the all the input 
    if not groq_api_key.strip() or not generic_url.strip():
        # st.write("Enter the required Input")
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
                    # headers= "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
                    headers= {})
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

