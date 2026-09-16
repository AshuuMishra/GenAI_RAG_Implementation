from langgraph.prebuilt import tools_condition

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"] = "TRUE"
os.environ["LANGSMITH_PROJECT"] = "Debugging_AI_AGENT"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"


llm = ChatGroq(
    model="openai/gpt-oss-20b"
)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def make_tool_graph():

    # Define the tool
    @tool
    def add(a: int, b: int):
        """Add two numbers a and b."""
        return a + b

    # Create ToolNode
    tool_node = ToolNode([add])

    # Bind tool to LLM
    llm_with_tool = llm.bind_tools([add])

    # LLM node
    def call_llm_tool(state: State):
        return {
            "messages": [
                llm_with_tool.invoke(state["messages"])
            ]
        }

    # Create graph
    graph_builder = StateGraph(State)

    # Add nodes
    graph_builder.add_node(
        "tool_calling_llm",
        call_llm_tool
    )

    graph_builder.add_node(
        "tools",
        tool_node
    )

    # START → LLM
    graph_builder.add_edge(
        START,
        "tool_calling_llm"
    )

    # LLM → either tool or END
    graph_builder.add_conditional_edges(
        "tool_calling_llm",
        tools_condition
    )

    # Tool → LLM
    graph_builder.add_edge(
        "tools",
        "tool_calling_llm"
    )

    # Compile
    graph = graph_builder.compile()

    return graph


tool_agent = make_tool_graph()
