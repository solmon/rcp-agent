"""LangGraph Studio compatible entry point for Recipe Agent."""

from typing import Literal, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from agent.state import RecipeAgentState
from agent.nodes import (
    classify_intent_node,
    llm_node,
    recipe_confirmation_node,
    recipe_plan_confirm_node,
    user_approval_node,
    tool_execution_node,
    recipe_execution_complete_node,
    shopping_cart_recommendation_node
)
from agent.tools import recipe_tools

# MCP client setup with error handling
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Global variables for MCP tools
_mcp_tools: List[BaseTool] = []

def get_mcp_tools() -> List[BaseTool]:
    """Get cached MCP tools."""
    return _mcp_tools.copy()

def route_intent(state: RecipeAgentState) -> Literal["search", "general"]:
    """Route based on classified intent."""
    intent = state.get("intent", "general")
    return intent

def create_studio_recipe_agent() -> StateGraph:
    """Create the Recipe Agent graph optimized for LangGraph Studio."""
    
    # Create the graph
    workflow = StateGraph(RecipeAgentState)
    
    # Add nodes for the enhanced workflow
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("recipe_llm", llm_node)
    workflow.add_node("recipe_confirmation", recipe_confirmation_node)
    workflow.add_node("recipe_plan_llm", llm_node)
    workflow.add_node("recipe_plan_confirmation", recipe_plan_confirm_node)    
    workflow.add_node("recipe_execution_llm", llm_node)
    workflow.add_node("user_approval", user_approval_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("recipe_execution_complete", recipe_execution_complete_node)
    workflow.add_node("shopping_cart_recommendation", shopping_cart_recommendation_node)
    
    # Set entry point
    workflow.set_entry_point("classify_intent")
    
    # WORKFLOW ROUTING:
    
    # 1. Intent classification -> Recipe search or general response
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "search": "recipe_llm",
            "general": END
        }
    )
    
    # 2. Recipe LLM -> Recipe confirmation (interrupt point)
    workflow.add_conditional_edges(
        "recipe_llm",
        lambda state: "recipe_confirmation" if state.get("recipes") else "end",
        {
            "recipe_confirmation": "recipe_confirmation",
            "end": END
        }
    )

    # Recipe confirmation -> User approval (interrupt point)
    workflow.add_edge("recipe_confirmation", "user_approval")

    # Recipe plan LLM -> Recipe plan confirmation (interrupt point)
    workflow.add_conditional_edges(
        "recipe_plan_llm",
        lambda state: "recipe_plan_confirmation" if state.get("plan_extract") else "end",
        {
            "recipe_plan_confirmation": "recipe_plan_confirmation",
            "end": END
        }
    )
    
    # Recipe plan confirmation -> User approval (interrupt point)
    workflow.add_edge("recipe_plan_confirmation", "user_approval")
    
    # User approval -> Route based on confirmation state
    workflow.add_conditional_edges(
        "user_approval",
        lambda state: (
            "recipe_plan_llm" if state.get("recipe_confirmed") and not state.get("recipe_plan_confirmed") else
            "recipe_execution_llm" if state.get("recipe_plan_confirmed") else
            "recipe_llm" if state.get("workflow_stage") == "recipe_search" else
            END
        ),
        {
            "recipe_plan_llm": "recipe_plan_llm",
            "recipe_execution_llm": "recipe_execution_llm", 
            "recipe_llm": "recipe_llm",
            "end": END
        }
    )

    # Recipe execution LLM -> Check for tool calls or completion
    workflow.add_conditional_edges(
        "recipe_execution_llm",
        lambda state: "tool_execution" if state.get("workflow_stage") == "tool_execution" else "recipe_execution_complete",
        {
            "tool_execution": "tool_execution",
            "recipe_execution_complete": "recipe_execution_complete"
        }
    )

    # Tool execution -> Recipe execution complete
    workflow.add_edge("tool_execution", "recipe_execution_complete")

    # Recipe execution complete -> Shopping cart recommendation
    workflow.add_conditional_edges(
        "recipe_execution_complete",
        lambda state: "shopping_cart_recommendation" if state.get("tool_outputs") else "end",
        {
            "shopping_cart_recommendation": "shopping_cart_recommendation",
            "end": END
        }
    )

    # Shopping cart recommendation -> END
    workflow.add_edge("shopping_cart_recommendation", END)
    
    # Compile without checkpointer for LangGraph Studio
    # LangGraph Studio handles persistence automatically
    return workflow.compile(interrupt_before=["user_approval"])

# Create the graph for LangGraph Studio
recipe_agent = create_studio_recipe_agent()

# Export the compiled graph for LangGraph Studio
__all__ = ["recipe_agent"]
