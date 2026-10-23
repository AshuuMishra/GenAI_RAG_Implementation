from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq

import asyncio

import os 
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

# intialising the Groq llm model
llm=ChatGroq(
    model="openai/gpt-oss-20b"
)

async def main():
    client=MultiServerMCPClient(
        {
            "Math":{
                "command":"python",
                "args":[r"C:\Users\CloudJournee\Desktop\python\python\langchain_update\GenAI_RAG_Implementation\mathserver.py"],
                "transport":"stdio",
            },
            "Weather":{

                "url":"http://localhost:8000/mcp",
                "transport":"streamable-http"
            }
        }
    )
    

    tools=await client.get_tools() 
    agent = create_react_agent(
        llm,tools
    )

    math_response = await agent.ainvoke(
        {"messages":[
            {
            "role":"user",
            "content" : "what's (3+(-1)*2.11)"
            }
                    ]
        }
    )

    print("math_response",math_response['messages'][-1].content)

    math_response = await agent.ainvoke(
        {
            "messages" :[
                {
                    "role" : "user",
                    "content" : "what the weather is Deoria uttar pradesh"
                }
            ]
        }
    )
      
    print("math_response",math_response['messages'][-1].content)

asyncio.run(main())
