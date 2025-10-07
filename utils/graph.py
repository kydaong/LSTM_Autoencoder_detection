# Created by aaronkueh on 9/19/2025
# aom/tools/graph.py

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from aom.tools.local_summarizer import LocalSummarizer

class SummarizeState(TypedDict, total=False):
    prompt: str
    max_new_tokens: int
    output: str

def make_summarize_node(model_name: str = "HuggingFaceTB/SmolLM3-3B"):
    slm = LocalSummarizer(model_name=model_name)
    def summarize_node(state: SummarizeState) -> SummarizeState:
        out = slm.summarize(state["prompt"], state.get("max_new_tokens", 640))
        return {**state, "output": out}
    return summarize_node

def make_summarize_app(model_name: str = "HuggingFaceTB/SmolLM3-3B"):
    summarize_node = make_summarize_node(model_name)
    graph = StateGraph(SummarizeState)
    graph.add_node("summarize", summarize_node)
    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()

