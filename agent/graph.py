"""Recipe Agent LangGraph Implementation."""

import asyncio
import logging
from typing import Literal, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
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
    logging.warning("langchain_mcp_adapters not available. Running without MCP tools.")

logger = logging.getLogger(__name__)

# Global variables for MCP tools
_mcp_client: Optional[any] = None
_mcp_tools: List[BaseTool] = []


async def initialize_mcp_client():
    """Initialize MCP client and load tools."""
    global _mcp_client, _mcp_tools
    
    if not MCP_AVAILABLE:
        logger.warning("MCP not available, skipping initialization")
        return []
    
    try:
        # Initialize MCP client
        _mcp_client = MultiServerMCPClient(
            {
                "grocery": {
                    "transport": "streamable_http",
                    "url": "http://localhost:8000/mcp/"
                },
            }
        )
        
        # Get tools asynchronously
        _mcp_tools = await _mcp_client.get_tools()
        logger.info(f"Loaded {len(_mcp_tools)} MCP tools")
        return _mcp_tools
        
    except Exception as e:
        logger.error(f"Failed to initialize MCP client: {e}")
        _mcp_tools = []
        return []


def get_mcp_tools() -> List[BaseTool]:
    """Get cached MCP tools."""
    return _mcp_tools.copy()


def route_intent(state: RecipeAgentState) -> Literal["search", "general"]:
    """Route based on classified intent."""
    intent = state.get("intent", "general")
    return intent

def should_call_tools(state: RecipeAgentState) -> Literal["tools", "cart_confirmation"]:
    """Check if grocery LLM wants to call tools."""
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
        return "tools"
    return "cart_confirmation"

def should_continue(state: RecipeAgentState) -> Literal["continue", "end"]:
    """Determine if processing should continue."""
    if state.get("processing_complete", False):
        return "end"
    if state.get("error_message"):
        return "end"
    return "continue"


def create_recipe_agent(include_mcp_tools: bool = False) -> StateGraph:
    
    
    """Create the Recipe Agent graph with enhanced workflow.
    
    Args:
        include_mcp_tools: Whether to include MCP tools in the agent
    """
    
    # Create the graph
    workflow = StateGraph(RecipeAgentState)
    
    # Add nodes for the enhanced workflow
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("recipe_llm", llm_node)  # Unified LLM node
    workflow.add_node("recipe_confirmation", recipe_confirmation_node)
    workflow.add_node("recipe_plan_llm", llm_node)  # Unified LLM node
    workflow.add_node("recipe_plan_confirmation", recipe_plan_confirm_node)    
    workflow.add_node("recipe_execution_llm", llm_node)  # Unified LLM node
    workflow.add_node("user_approval", user_approval_node)  # New approval node
    workflow.add_node("tool_execution", tool_execution_node)  # New tool execution node
    workflow.add_node("recipe_execution_complete", recipe_execution_complete_node)  # New completion node
    workflow.add_node("shopping_cart_recommendation", shopping_cart_recommendation_node)  # New shopping cart node
        
    # Prepare tools
    all_tools = recipe_tools.copy()

    # Add MCP tools if requested and available
    if include_mcp_tools:
        mcp_tools = get_mcp_tools()
        if mcp_tools:
            all_tools.extend(mcp_tools)
            logger.info(f"Added {len(mcp_tools)} MCP tools to agent")
        else:
            logger.warning("MCP tools requested but none available")

    
    # Set entry point
    workflow.set_entry_point("classify_intent")
    
    # WORKFLOW ROUTING:
    
    # 1. Intent classification -> Recipe search or general response
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "search": "recipe_llm",
            "general": END  # Use recipe LLM for general queries too
        }
    )
    
    # 3. Recipe LLM -> Recipe confirmation (interrupt point)
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
    
    # Create checkpointer for state persistence during interrupts
    checkpointer = MemorySaver()
    
    # Compile the graph with interrupts at approval points and checkpointer
    return workflow.compile(interrupt_before=["user_approval"], checkpointer=checkpointer)


# Create the default agent instance (without MCP tools for now)
recipe_agent = create_recipe_agent()


async def create_recipe_agent_with_mcp() -> StateGraph:
    """Create the Recipe Agent graph with MCP tools loaded."""
    try:
        # Initialize MCP tools first
        await initialize_mcp_client()
        # Create agent with MCP tools
        agent = create_recipe_agent(include_mcp_tools=True)
        logger.info("Successfully created recipe agent with MCP tools")
        return agent
    except Exception as e:
        logger.error(f"Failed to create agent with MCP tools: {e}")
        logger.info("Falling back to agent without MCP tools")
        return create_recipe_agent(include_mcp_tools=False)


async def run_recipe_agent_with_mcp(query: str, **kwargs) -> dict:
    """Run the recipe agent with a query (including MCP tools)."""
    initial_state = {
        "user_query": query,
        "messages": [HumanMessage(content=query)],
        "recipes": [],
        "needs_user_input": False,
        "recipe_confirmed": False,
        "workflow_stage": "initial",  # Start with initial instead of None
        "processing_complete": False,
        "tool_outputs": {},
        "pending_tool_calls": []  # Initialize pending tool calls
    }
    
    # Create agent with MCP tools
    agent = await create_recipe_agent_with_mcp()
    # Run the agent asynchronously
    return await agent.ainvoke(initial_state)
