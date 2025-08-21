import os
import sys

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


import json
import logging
import math
import time
from typing import Dict

from langgraph.checkpoint.postgres import PostgresSaver
from markdownify import markdownify
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import SecretStr, BaseModel
from typing_extensions import Literal, Final, Any

from src.configuration import Configuration

from src.utilities import strip_thinking_tokens, get_config_value, duckduckgo_search, deduplicate_and_format_sources, \
    format_sources
from src.prompts.query_writer_instructions import get_current_date, query_writer_instructions
from src.prompts.summarizer_instructions import summarizer_instructions
from src.prompts.reflection_instructions import reflection_instructions
from src.prompts.editor_instructions import editor_instructions
from src.states.summary_state import SummaryState, SummaryStateInput, SummaryStateOutput


def generate_query(state: SummaryState, config: RunnableConfig):
    current_date = get_current_date()

    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=state.topic
    )

    configurable = Configuration.from_runnable_config(config)

    llm_json_mode = ChatOpenAI(
        base_url=configurable.base_url,
        model=configurable.local_llm,
        api_key=SecretStr("None"),
        temperature=0
    )
    llm_json_mode.with_structured_output(method="json_mode")

    result = llm_json_mode.invoke(
        [SystemMessage(content=formatted_prompt),
        HumanMessage(content=f"Generate a query for web search:")]
    )

    content = result.content

    try:
        stripped_json = strip_thinking_tokens(content).strip()
        query = json.loads(stripped_json)
        search_query = query['query']
    except (json.JSONDecodeError, KeyError):
        logging.error(f"Failed to parse JSON response: {content}")
        # If parsing fails or the key is not found, use a fallback query
        if configurable.strip_thinking_tokens:
            content = strip_thinking_tokens(content).strip()
        search_query = content
    return {"query": search_query, "loop_count": 0, "sources": []}




def web_research(state: SummaryState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    # Configure
    search_results = duckduckgo_search(state.query, max_results=3,
                                           fetch_full_page=configurable.fetch_full_page)
    search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000,
                                                    fetch_full_page=configurable.fetch_full_page)
    return {"sources": [format_sources(search_results)], "loop_count": state.loop_count + 1,
            "web_resultset": [search_str]}


def summarize_sources(state: SummaryState, config: RunnableConfig):
    """LangGraph node that summarizes web research results.

    Uses an LLM to create or update a running summary based on the newest web research
    results, integrating them with any existing summary.

    Args:
        state: Current graph state containing research topic, running summary,
              and web research results
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including running_summary key containing the updated summary
    """

    # Existing summary
    existing_summary = state.summary

    # Most recent web research
    most_recent_web_research = state.web_resultset[-1]

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Context> \n {most_recent_web_research} \n <New Context>"
            f"Update the Existing Summary with the New Context on this topic: \n <User Input> \n {state.topic} \n <User Input>\n\n"
        )
    else:
        human_message_content = (
            f"<Context> \n {most_recent_web_research} \n <Context>"
            f"Create a Summary using the Context on this topic: \n <User Input> \n {state.topic} \n <User Input>\n\n"
        )

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)

    # Default to Ollama
    llm = ChatOpenAI(
        base_url=configurable.base_url,
        model=configurable.local_llm,
        api_key=SecretStr("None"),
        temperature=0
    )
    #llm.with_structured_output(method="json_mode")
    result = llm.invoke(
        [SystemMessage(content=summarizer_instructions),
         HumanMessage(content=human_message_content)]
    )

    # Strip thinking tokens if configured
    running_summary = result.content
    if configurable.strip_thinking_tokens:
        running_summary = strip_thinking_tokens(running_summary)

    return {"summary": running_summary}


def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    # Generate a query
    configurable = Configuration.from_runnable_config(config)

    # Default to Ollama
    llm_json_mode = ChatOpenAI(
        base_url=configurable.base_url,
        model=configurable.local_llm,
        api_key=SecretStr("None"),
        temperature=0
    )
    llm_json_mode.with_structured_output(method="json_mode")


    result = llm_json_mode.invoke(
        [SystemMessage(content=reflection_instructions.format(research_topic=state.topic)),
         HumanMessage(
             content=f"Reflect on our existing knowledge: \n === \n {state.summary}, \n === \n And now identify a knowledge gap and generate a follow-up web search query:")]
    )

    # Strip thinking tokens if configured
    try:
        # Try to parse as JSON first
        reflection_content = json.loads(result.content)
        # Get the follow-up query
        query = reflection_content.get('follow_up_query')
        # Check if query is None or empty
        if not query:
            # Use a fallback query
            return {"query": f"Tell me more about {state.topic}"}
        return {"query": query}
    except (json.JSONDecodeError, KeyError, AttributeError):
        # If parsing fails or the key is not found, use a fallback query
        return {"query": f"Tell me more about {state.topic}"}


def finalize_summary(state: SummaryState):
    # Deduplicate sources before joining
    seen_sources = set()
    unique_sources = []

    for source in state.sources:
        # Split the source into lines and process each individually
        for line in source.split('\n'):
            # Only process non-empty lines
            if line.strip() and line not in seen_sources:
                seen_sources.add(line)
                unique_sources.append(line)

    # Join the deduplicated sources
    all_sources = "\n".join(unique_sources)
    state.summary = f"## Summary\n{state.summary}\n\n ### Sources:\n{all_sources}"
    return {"summary": state.summary}

def edit_summary(state: SummaryState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)

    llm_json_mode = ChatOpenAI(
        base_url=configurable.base_url,
        model=configurable.local_llm,
        api_key=SecretStr("None"),
        temperature=0
    )

    result = llm_json_mode.invoke(
        [SystemMessage(content=editor_instructions.format(research_topic=state.topic)),
         HumanMessage(
             content=f"Edit on our existing summary: \n === \n {state.summary}, \n === \n ")]
    )

    return {"summary": strip_thinking_tokens(result.content)}

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "web_research"]:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_web_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """

    configurable = Configuration.from_runnable_config(config)
    if state.loop_count <= configurable.max_web_research_loops:
        return "web_research"
    else:
        return "finalize_summary"



# Constants for node names (avoid magic strings)
NODE_GENERATE_QUERY: Final[str] = "generate_query"
NODE_WEB_RESEARCH: Final[str] = "web_research"
NODE_SUMMARIZE_SOURCES: Final[str] = "summarize_sources"
NODE_REFLECT_ON_SUMMARY: Final[str] = "reflect_on_summary"
NODE_FINALIZE_SUMMARY: Final[str] = "finalize_summary"
NODE_EDIT_SUMMARY: Final[str] = "edit_summary"

def build_summary_graph() -> Any:
    """
    Build and compile the summary research graph.
    """
    builder = StateGraph(
        SummaryState,
        input_schema=SummaryStateInput,
        output_schema=SummaryStateOutput,
        config_schema=Configuration,
    )

    # Nodes
    builder.add_node(NODE_GENERATE_QUERY, generate_query)
    builder.add_node(NODE_WEB_RESEARCH, web_research)
    builder.add_node(NODE_SUMMARIZE_SOURCES, summarize_sources)
    builder.add_node(NODE_REFLECT_ON_SUMMARY, reflect_on_summary)
    builder.add_node(NODE_FINALIZE_SUMMARY, finalize_summary)
    builder.add_node(NODE_EDIT_SUMMARY, edit_summary)

    # Edges
    builder.add_edge(START, NODE_GENERATE_QUERY)
    builder.add_edge(NODE_GENERATE_QUERY, NODE_WEB_RESEARCH)
    builder.add_edge(NODE_WEB_RESEARCH, NODE_SUMMARIZE_SOURCES)
    builder.add_edge(NODE_SUMMARIZE_SOURCES, NODE_REFLECT_ON_SUMMARY)
    builder.add_conditional_edges(NODE_REFLECT_ON_SUMMARY, route_research)
    builder.add_edge(NODE_FINALIZE_SUMMARY, NODE_EDIT_SUMMARY)
    builder.add_edge(NODE_EDIT_SUMMARY, END)

    return builder.compile()


def get_run_config() -> Dict[str, Dict[str, int]]:
    """
    Create a runnable config with a unique thread id.
    """
    return {"configurable": {"thread_id": int(time.time())}}


def prompt_for_topic(prompt: str = "Give me a topic to research: ") -> str:
    """
    Get a research topic from the user.
    """
    return input(prompt).strip()


def main() -> None:
    """
    Orchestrate graph setup and execution.
    """

    graph = build_summary_graph()

    user_topic = prompt_for_topic()
    config = get_run_config()

    result = graph.invoke({"topic": user_topic}, config)
    content: str = markdownify(result['summary'])
    thread_id: str = config["configurable"]["thread_id"]
    with open(f"summary_{user_topic}_{thread_id}.md", "w", encoding="utf-8") as file:
        file.write(content)

# Entry point
if __name__ == "__main__":
    main()
