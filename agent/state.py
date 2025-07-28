"""Recipe Agent State definition."""

from typing import Dict, List, Optional, TypedDict, Annotated, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RecipeAgentState(TypedDict):
    """Enhanced State for the Recipe Agent with Interrupt Support."""
    
    # Conversation history
    messages: Annotated[List[BaseMessage], add_messages]
    display_messages: List[str]  # For displaying in UI
    
    # User query and intent
    user_query: Optional[str]
    human_response: Optional[str]  # For capturing user responses during interrupts
    intent: Optional[str]  # search, recommend, substitute, plan, etc.
    
    # Recipe-related data    
    recipes: List[Dict]
    selected_recipe: Optional[Dict]  # Recipe selected by user    

    plan_extract: Optional[str]  # Extracted plan from recipe article

    # ingredients: List[str]
    searched_ingredients: List[Dict]  # Ingredients found via MCP search
    ingredients_to_cart: List[Dict]   # Confirmed ingredients for cart
    
    # Search and recommendation data
    # search_results: List[Dict]
        
    # Processing flags    
    recipe_confirmed: Optional[bool]   # User confirmed ingredients for grocery search
    recipe_plan_confirmed: Optional[bool]   # User confirmed ingredients for grocery search

    grocery_items_confirmed: Optional[bool]  # User confirmed grocery items for cart
    workflow_stage: Optional[str]  # "recipe_search", "ingredient_confirmation", "grocery_search", "cart_confirmation"
    error_message: Optional[str]
    processing_complete: bool
    
    # Tool outputs
    tool_outputs: Dict[str, any]
    pending_tool_calls: List[Dict]  # Tool calls waiting to be executed
    
    # Shopping cart data with enhanced structure
    shopping_cart_data: Optional[Dict[str, Any]]  # Comprehensive shopping cart recommendations with categorized items
   

