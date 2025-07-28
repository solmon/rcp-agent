"""Recipe Agent Graph Nodes."""
import os
import sys
import asyncio
from dotenv import load_dotenv

from typing import Dict, Any

import re
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import RecipeAgentState
from agent.tools import recipe_tools
from agent.tools import mock_recipes
# context7 prompt imports
from prompts.system_prompts import RECIPE_SYSTEM_PROMPT, GROCERY_SYSTEM_PROMPT, RECIPE_PLAN_SYSTEM_PROMPT, GROCERY_EXEC_SYSTEM_PROMPT
from prompts.chat_prompts import RECIPE_CHAT_PROMPT, GROCERY_CHAT_PROMPT, RECIPE_ARTICLE_CHAT_PROMPT, GROCERY_EXEC_CHAT_PROMPT

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


def stream_message(state: RecipeAgentState, message: str):
    """Stream a message to the UI (currently just prints)."""
    # print(f"📢 {message}")


def classify_intent_node(state: RecipeAgentState) -> Dict[str, Any]:
    """Classify the user's intent from their query."""
    user_query = state.get("user_query", "")
    
    # Simple intent classification logic
    query_lower = user_query.lower()
    
    if any(word in query_lower for word in ["search", "find", "recipe for", "how to make"]):
        intent = "search"
    else:
        intent = "general"
    
    return {
        "intent": intent,
        "workflow_stage": "initial",
        "display_messages": [f"🔍 Understanding your request: '{user_query}'"]
    }

# --- Modularized LLM Node Logic ---
def _select_llm_prompts_and_tools(state: RecipeAgentState):
    """Selects the correct prompt templates and tools for the current workflow stage."""
    
    from agent.graph import get_mcp_tools
    workflow_stage = state.get("workflow_stage", "recipe_llm")
    user_query = state.get("user_query", "")
    context_parts = []
    all_tools = recipe_tools.copy()

    if workflow_stage in ["recipe_execution"]:
        mcp_tools = get_mcp_tools()
        if mcp_tools:
            all_tools.extend(mcp_tools)
        
        # recipe_plan = state.get("plan_extract", {})
        recipe_plan=state.get("selected_recipe", {})
        tool_names = ", ".join([tool.name for tool in all_tools])
        num_tools = len(all_tools)
        rendered_prompt = [
            GROCERY_EXEC_SYSTEM_PROMPT.format(num_tools=num_tools, tool_names=tool_names),
            GROCERY_EXEC_CHAT_PROMPT.format(user_query=recipe_plan)
        ]
        return rendered_prompt, all_tools, "execute"
    elif workflow_stage in ["recipe_planning"]:
        mcp_tools = get_mcp_tools()
        if mcp_tools:
            all_tools.extend(mcp_tools)
        tool_names = ", ".join([tool.name for tool in all_tools])
        num_tools = len(all_tools)
        context = " | ".join(context_parts) if context_parts else "No specific context"
        rendered_prompt = [
            RECIPE_PLAN_SYSTEM_PROMPT.format(num_tools=num_tools, tool_names=tool_names),
            RECIPE_ARTICLE_CHAT_PROMPT.format(selected_recipe=state.get("selected_recipe", {}), context=context)
        ]
        return rendered_prompt, all_tools, "plan"
    else:        
        context = " | ".join(context_parts) if context_parts else "No specific context"
        rendered_prompt = [
            RECIPE_SYSTEM_PROMPT.format(),
            RECIPE_CHAT_PROMPT.format(user_query=user_query, context=context)
        ]
        return rendered_prompt, all_tools, "recipe"

async def _invoke_llm_with_tools(llm, rendered_prompt, tools=None):
    """Invoke the LLM, binding tools if provided."""
    if tools:
        llm = llm.bind_tools(tools)        
        return llm.invoke(rendered_prompt)                
    else:
        # No tools, just invoke with the prompt        
        return llm.invoke(rendered_prompt)       

def _extract_recipes_from_response(response):
    """Extract recipes from LLM response content."""
    recipes = []
    if response and hasattr(response, "content"):
        content = response.content
        # Try to extract title from the first line or bolded heading
        title_match = re.search(r"^(.*?recipe.*?)\n", content, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()
        else:
            # Fallback: use first non-empty line
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            title = lines[0] if lines else "Recipe"
        
        recipes.append({
            "title": title,
            "recipe_msg": response.content        
        })
    return recipes

async def llm_node(state: RecipeAgentState) -> Dict[str, Any]:
    """Unified and modular LLM node for both recipe and grocery workflow stages."""
    # Select prompts and tools
    rendered_prompt, all_tools, mode = _select_llm_prompts_and_tools(state)
    user_query = state.get("user_query", "")
    # LLM setup
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
        transport="rest",
        client_options={"api_endpoint": "https://generativelanguage.googleapis.com"}
    )
    try:
        # Mock for recipe mode
        if mode == "recipe":
            is_mock_recipes = False
            if is_mock_recipes:
                state["recipes"] = mock_recipes()
                
                return {
                    "messages": [],
                    "recipes": state["recipes"],
                    "workflow_stage": "recipe_display",
                    "display_messages": ["🍳 Here are some mock recipes for testing!"]
                }
        
        # LLM invocation
        response = await _invoke_llm_with_tools(llm, rendered_prompt, all_tools if mode in ["plan","execute"] else None)
        # print(f"LLM Response: {getattr(response, 'content', response)}")
        messages = state.get("messages", [])
        
        if mode == "plan":
            messages.append(AIMessage(content=response.content))
            plan_extract = response.content
            return {
                "messages": messages,
                "workflow_stage": "recipe_plan_display",
                "plan_extract": plan_extract,
                "recipe_plan_confirmed": False,
                "display_messages": [response.content]
            }
        elif mode == "execute":
            messages.append(AIMessage(content=response.content))
            
            # Check if there are tool calls to execute
            if hasattr(response, 'tool_calls') and response.tool_calls:
                return {
                    "messages": messages,
                    "workflow_stage": "tool_execution",
                    "pending_tool_calls": response.tool_calls,
                    "tool_outputs": {},
                    "display_messages": [response.content]
                }
            else:
                # No tool calls, return response directly
                return {
                    "messages": messages,
                    "workflow_stage": "recipe_execution_complete",
                    "tool_outputs": {"response": response.content},
                    "display_messages": [response.content]
                }
        else:
            messages.append(AIMessage(content=response.content))
            recipes = _extract_recipes_from_response(response)
            if recipes:
                state["recipes"] = recipes
            return {
                "messages": messages,
                "recipes": state["recipes"],
                "workflow_stage": "recipe_display",
                "display_messages": [response.content]
            }

    except Exception as e:
        error_msg = f"Error in LLM node: {str(e)}"
        return {
            "error_message": error_msg,
            "processing_complete": True,
            "display_messages": [f"❌ {error_msg}"]
        }


def recipe_confirmation_node(state: RecipeAgentState) -> Dict[str, Any]:
    """Display ingredients to user for confirmation before grocery search."""
    # Clear previous display messages
    display_messages = []
    
    recipes = state.get("recipes", [])
    if not recipes:
        return {
            "error_message": "No recipes available for ingredient confirmation",
            "processing_complete": True,
            "display_messages": display_messages
        }

    selected_recipe = state.get("selected_recipe", recipes[0] if recipes else {})
    
    # Context7: Enhanced UI-friendly confirmation message
    confirmation_message = f"""🍽️ **{selected_recipe.get('title', 'Recipe')}**

{selected_recipe.get('recipe_msg', 'No recipe details available')}

💡 **Next Steps:**
I can help you create a detailed cooking plan and then search for ingredients in local grocery stores!

**Would you like me to proceed with creating a cooking plan? (yes/no)**
"""
    
    # Add to display messages for UI
    display_messages.append(confirmation_message)

    return {
        "selected_recipe": selected_recipe,
        "workflow_stage": "recipe_confirmation", 
        "recipe_confirmed": False,
        "display_messages": display_messages,
        "processing_complete": False
    }

def recipe_plan_confirm_node(state: RecipeAgentState) -> Dict[str, Any]:
    """Display cooking plan to user for confirmation before execution."""
    # Clear previous display messages
    display_messages = []
    
    plan_extract = state.get("plan_extract", "")
    if not plan_extract:
        return {
            "error_message": "No cooking plan available for display",
            "processing_complete": True,
            "display_messages": display_messages
        }

    selected_recipe = state.get("selected_recipe", {})
    
    # Context7: Enhanced UI-friendly plan confirmation
    confirmation_message = f"""📋 **Cooking Plan for {selected_recipe.get('title', 'Recipe')}**

{plan_extract}

🛒 **Ready to start cooking?**
I'll search for the best ingredients and help you execute this plan step by step!

**Would you like me to proceed with the cooking execution? (yes/no)**
"""
    
    # Add to display messages for UI
    display_messages.append(confirmation_message)
    
    return {
        "selected_recipe": selected_recipe,
        "plan_extract": plan_extract,
        "workflow_stage": "recipe_plan_confirmation", 
        "recipe_plan_confirmed": False,
        "display_messages": display_messages,
        "processing_complete": False
    }


def user_approval_node(state: RecipeAgentState) -> Dict[str, Any]:
    """Handle user approval responses during interrupts."""
    # Clear previous display messages at start
    display_messages = []
    
    workflow_stage = state.get("workflow_stage", "")
    human_response = state.get("human_response", "").lower()
    
    if workflow_stage == "recipe_confirmation":
        if human_response in ["yes", "proceed", "continue", "plan", "cook"]:
            return {
                "recipe_confirmed": True,
                "recipe_plan_confirmed": True,
                "workflow_stage": "recipe_execution",
                "display_messages": display_messages,
                "processing_complete": False
            }
        elif human_response in ["no", "back", "change", "different", "new"]:
            return {
                "workflow_stage": "recipe_search",
                "recipe_confirmed": False,
                "display_messages": display_messages,
                "processing_complete": False
            }
        else:
            # Invalid response, ask again
            display_messages.append("Please respond with 'yes' to proceed or 'no' to go back.")
            return {
                "workflow_stage": "recipe_confirmation",
                "display_messages": display_messages,
                "processing_complete": False
            }
    
    elif workflow_stage == "recipe_plan_confirmation":
        if human_response in ["yes", "proceed", "execute", "start", "cook", "continue"]:
            return {
                "recipe_plan_confirmed": True,
                "workflow_stage": "recipe_execution",
                "display_messages": display_messages,
                "processing_complete": False
            }
        elif human_response in ["no", "back", "change", "modify"]:
            return {
                "workflow_stage": "recipe_planning",
                "recipe_plan_confirmed": False,
                "display_messages": display_messages,
                "processing_complete": False
            }
        else:
            # Invalid response, ask again
            display_messages.append("Please respond with 'yes' to proceed or 'no' to go back.")
            return {
                "workflow_stage": "recipe_plan_confirmation",
                "display_messages": display_messages,
                "processing_complete": False
            }
    
    # Default fallback
    return {
        "display_messages": display_messages,
        "processing_complete": False
    }


def tool_execution_node(state: RecipeAgentState) -> Dict[str, Any]:
    """Execute tool calls from the LLM and store results in tool_outputs."""
    import asyncio
    from agent.graph import get_mcp_tools
    from agent.tools import recipe_tools
    
    display_messages = []
    tool_outputs = state.get("tool_outputs", {})
    
    # Get pending tool calls
    pending_tool_calls = state.get("pending_tool_calls", [])
    if not pending_tool_calls:
        return {
            "error_message": "No pending tool calls to execute",
            "processing_complete": True,
            "display_messages": display_messages
        }
    
    # Prepare all available tools
    all_tools = recipe_tools.copy()
    mcp_tools = get_mcp_tools()
    if mcp_tools:
        all_tools.extend(mcp_tools)
    
    # Create a mapping of tool names to tool objects
    tool_map = {tool.name: tool for tool in all_tools}
    
    async def execute_tool_async(tool_instance, tool_args):
        """Execute tool asynchronously."""
        if hasattr(tool_instance, 'ainvoke'):
            return await tool_instance.ainvoke(tool_args)
        else:
            # Fallback to sync invoke
            return tool_instance.invoke(tool_args)
    
    def execute_tools_sync():
        """Execute all tools and return results."""
        results = {}
        messages = []
        
        # Create event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def execute_all_tools():
            for tool_call in pending_tool_calls:
                # Handle different tool_call formats (LangChain vs direct dict)
                if hasattr(tool_call, 'name'):
                    # LangChain ToolCall object
                    tool_name = tool_call.name
                    tool_args = tool_call.args
                    tool_id = getattr(tool_call, 'id', f"call_{tool_name}")
                elif isinstance(tool_call, dict):
                    # Dictionary format
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", f"call_{tool_name}")
                else:
                    # Skip unknown format
                    continue
                
                if tool_name in tool_map:
                    try:
                        # Execute the tool
                        tool_instance = tool_map[tool_name]
                        result = await execute_tool_async(tool_instance, tool_args)
                        
                        # Store the result
                        results[tool_id] = {
                            "tool_name": tool_name,
                            "args": tool_args,
                            "result": result,
                            "status": "success"
                        }
                        
                        messages.append(f"✅ Executed {tool_name}: {str(result)[:100]}...")
                        
                    except Exception as e:
                        # Store the error
                        results[tool_id] = {
                            "tool_name": tool_name,
                            "args": tool_args,
                            "result": None,
                            "error": str(e),
                            "status": "error"
                        }
                        
                        messages.append(f"❌ Error executing {tool_name}: {str(e)}")
                else:
                    # Tool not found
                    results[tool_id] = {
                        "tool_name": tool_name,
                        "args": tool_args,
                        "result": None,
                        "error": f"Tool '{tool_name}' not found",
                        "status": "error"
                    }
                    
                    messages.append(f"❌ Tool '{tool_name}' not found")
        
        # Run the async function
        if loop.is_running():
            # If event loop is already running, use run_in_executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, execute_all_tools())
                future.result()
        else:
            loop.run_until_complete(execute_all_tools())
        
        return results, messages
    
    try:
        # Execute all tools
        tool_results, tool_messages = execute_tools_sync()
        tool_outputs.update(tool_results)
        display_messages.extend(tool_messages)
        
        # Check if we need to continue with more LLM processing or complete
        successful_executions = [output for output in tool_outputs.values() if output.get("status") == "success"]
        
        if successful_executions:
            # Tools executed successfully, continue to completion
            return {
                "tool_outputs": tool_outputs,
                "workflow_stage": "recipe_execution_complete",
                "display_messages": display_messages,
                "pending_tool_calls": [],  # Clear pending calls
                "processing_complete": False
            }
        else:
            # All tools failed, complete with error
            return {
                "tool_outputs": tool_outputs,
                "error_message": "All tool executions failed",
                "display_messages": display_messages,
                "pending_tool_calls": [],  # Clear pending calls
                "processing_complete": True
            }
            
    except Exception as e:
        return {
            "error_message": f"Error in tool execution node: {str(e)}",
            "tool_outputs": tool_outputs,
            "display_messages": display_messages,
            "processing_complete": True
        }


def recipe_execution_complete_node(state: RecipeAgentState) -> Dict[str, Any]:
    """Final node to summarize the recipe execution process."""
    display_messages = []
    tool_outputs = state.get("tool_outputs", {})
    
    if tool_outputs:
        # Summarize tool execution results
        successful_tools = [output for output in tool_outputs.values() if output.get("status") == "success"]
        failed_tools = [output for output in tool_outputs.values() if output.get("status") == "error"]
        
        summary_message = f"""🎉 **Recipe Execution Complete!**

✅ **Successfully executed:** {len(successful_tools)} tools
❌ **Failed:** {len(failed_tools)} tools

**Results Summary:**
"""
        
        for tool_id, output in tool_outputs.items():
            if output.get("status") == "success":
                summary_message += f"\n✅ {output['tool_name']}: {str(output['result'])[:100]}..."
            else:
                summary_message += f"\n❌ {output['tool_name']}: {output.get('error', 'Unknown error')}"
        
        summary_message += "\n\n🛒 **Generating shopping cart recommendations...**"
        display_messages.append(summary_message)
        
        # Don't mark as complete if we have tool outputs - let it flow to shopping cart
        return {
            "display_messages": display_messages,
            "processing_complete": False,
            "workflow_stage": "recipe_execution_summary"
        }
    else:
        display_messages.append("🎉 **Recipe process completed!**")
        return {
            "display_messages": display_messages,
            "processing_complete": True,
            "workflow_stage": "complete"
        }


def shopping_cart_recommendation_node(state: RecipeAgentState) -> Dict[str, Any]:
    """Analyze tool call responses and recommend a shopping cart."""
    display_messages = []
    tool_outputs = state.get("tool_outputs", {})
    selected_recipe = state.get("selected_recipe", {})
    plan_extract = state.get("plan_extract", "")
    
    if not tool_outputs:
        return {
            "error_message": "No tool outputs available for shopping cart recommendation",
            "processing_complete": True,
            "display_messages": display_messages
        }
    
    # Analyze tool responses to extract shopping information
    shopping_items = []
    store_recommendations = []
    price_comparisons = []
    ingredient_details = []
    nutritional_info = []
    
    def _extract_ingredients_from_text(text):
        """Extract ingredient information from text using patterns."""
        ingredients = []
        lines = text.split('\n')
        
        # Common ingredient patterns
        ingredient_patterns = [
            r'(\d+(?:\.\d+)?\s*(?:cups?|tbsp|tsp|lbs?|oz|grams?|kg|pounds?|ounces?|tablespoons?|teaspoons?)?)\s+(.+)',
            r'([^-•*\n]+?)(?:\s*[-–—]\s*\$?\d+\.?\d*)?$',
            r'•\s*(.+?)(?:\s*\$\d+\.?\d*)?$',
            r'-\s*(.+?)(?:\s*\$\d+\.?\d*)?$'
        ]
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue
                
            # Skip obvious non-ingredient lines
            skip_keywords = ['instructions', 'directions', 'steps', 'method', 'preparation', 'note:', 'tip:']
            if any(keyword in line.lower() for keyword in skip_keywords):
                continue
            
            # Try each pattern
            for pattern in ingredient_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    if len(match.groups()) == 2:
                        # Quantity and ingredient
                        quantity, ingredient = match.groups()
                        ingredients.append(f"{quantity.strip()} {ingredient.strip()}")
                    else:
                        # Just ingredient
                        ingredients.append(match.group(1).strip())
                    break
            else:
                # If no pattern matches, check if it looks like an ingredient
                if any(food_word in line.lower() for food_word in ['chicken', 'beef', 'pork', 'fish', 'vegetables', 'onion', 'garlic', 'salt', 'pepper', 'oil', 'butter', 'flour', 'sugar', 'rice', 'pasta', 'cheese', 'milk', 'eggs', 'tomato']):
                    ingredients.append(line)
        
        return ingredients
    
    def _extract_stores_from_text(text):
        """Extract store information from text."""
        stores = []
        lines = text.split('\n')
        
        store_keywords = ['market', 'grocery', 'store', 'supermarket', 'shop', 'mart', 'fresh', 'organic', 'whole foods', 'kroger', 'walmart', 'target', 'safeway', 'publix']
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in store_keywords):
                # Clean up store name
                store_name = re.sub(r'[•\-*]', '', line).strip()
                if store_name and len(store_name) > 2:
                    stores.append(store_name)
        
        return stores
    
    def _extract_prices_from_text(text):
        """Extract price information from text."""
        prices = []
        # Pattern to match prices
        price_pattern = r'\$\d+\.?\d*|\d+\.?\d*\s*dollars?|\d+\.?\d*\s*cents?'
        
        matches = re.findall(price_pattern, text, re.IGNORECASE)
        for match in matches:
            prices.append(match.strip())
        
        return prices
    
    # Process each tool output
    for tool_id, output in tool_outputs.items():
        if output.get("status") == "success":
            tool_name = output.get("tool_name", "")
            result = output.get("result", "")
            
            # Convert result to string if it's not already
            if isinstance(result, dict):
                result_text = str(result)
                # Try to extract structured data first
                if 'items' in result:
                    shopping_items.extend(result.get('items', []))
                if 'stores' in result:
                    store_recommendations.extend(result.get('stores', []))
                if 'prices' in result:
                    price_comparisons.extend(result.get('prices', []))
            elif isinstance(result, list):
                result_text = '\n'.join(str(item) for item in result)
                shopping_items.extend(result)
            else:
                result_text = str(result)
            
            # Extract information based on tool type and content
            if "search" in tool_name.lower() or "find" in tool_name.lower() or "ingredient" in tool_name.lower():
                # Extract ingredients from search results
                extracted_ingredients = _extract_ingredients_from_text(result_text)
                shopping_items.extend(extracted_ingredients)
                
                # Extract stores
                extracted_stores = _extract_stores_from_text(result_text)
                store_recommendations.extend(extracted_stores)
                
                # Extract prices
                extracted_prices = _extract_prices_from_text(result_text)
                price_comparisons.extend(extracted_prices)
            
            elif "price" in tool_name.lower() or "compare" in tool_name.lower() or "cost" in tool_name.lower():
                # Handle price comparison results
                extracted_prices = _extract_prices_from_text(result_text)
                price_comparisons.extend(extracted_prices)
                
                # Also check for store mentions in price comparisons
                extracted_stores = _extract_stores_from_text(result_text)
                store_recommendations.extend(extracted_stores)
            
            elif "store" in tool_name.lower() or "shop" in tool_name.lower():
                # Store-specific results
                extracted_stores = _extract_stores_from_text(result_text)
                store_recommendations.extend(extracted_stores)
            
            elif "nutrition" in tool_name.lower() or "health" in tool_name.lower():
                # Nutritional information
                nutritional_info.append(result_text)
    
    # Also extract ingredients from the recipe plan if available
    if plan_extract:
        plan_ingredients = _extract_ingredients_from_text(plan_extract)
        shopping_items.extend(plan_ingredients)
    
    # Extract ingredients from the recipe itself
    if selected_recipe.get('recipe_msg'):
        recipe_ingredients = _extract_ingredients_from_text(selected_recipe['recipe_msg'])
        shopping_items.extend(recipe_ingredients)
    
    # Extract ingredients from the recipe itself
    if selected_recipe.get('recipe_msg'):
        recipe_ingredients = _extract_ingredients_from_text(selected_recipe['recipe_msg'])
        shopping_items.extend(recipe_ingredients)
    
    # Clean and deduplicate the collected data
    def _clean_and_dedupe(items):
        """Clean and deduplicate a list of items."""
        cleaned = []
        seen = set()
        for item in items:
            if isinstance(item, str):
                # Clean the item
                cleaned_item = re.sub(r'[•\-*]', '', item).strip()
                cleaned_item = re.sub(r'\s+', ' ', cleaned_item)  # Remove extra spaces
                
                # Skip if too short or already seen
                if len(cleaned_item) > 2 and cleaned_item.lower() not in seen:
                    cleaned.append(cleaned_item)
                    seen.add(cleaned_item.lower())
        return cleaned
    
    # Clean and limit the data
    shopping_items = _clean_and_dedupe(shopping_items)[:15]  # Limit to 15 items
    store_recommendations = _clean_and_dedupe(store_recommendations)[:8]  # Limit to 8 stores
    price_comparisons = _clean_and_dedupe(price_comparisons)[:10]  # Limit to 10 price entries
    
    # Generate comprehensive shopping cart recommendation
    recipe_title = selected_recipe.get('title', 'Your Recipe')
    
    cart_recommendation = f"""🛒 **Smart Shopping Cart for {recipe_title}**

Based on the analysis of tool responses and your recipe, here's your personalized shopping recommendation:

"""
    
    # Add shopping items section
    if shopping_items:
        cart_recommendation += "**📋 Shopping List:**\n"
        
        # Categorize items if possible
        produce_items = []
        pantry_items = []
        protein_items = []
        dairy_items = []
        other_items = []
        
        produce_keywords = ['onion', 'garlic', 'tomato', 'pepper', 'carrot', 'celery', 'potato', 'lettuce', 'spinach', 'herb', 'lemon', 'lime', 'apple', 'banana']
        pantry_keywords = ['flour', 'sugar', 'salt', 'pepper', 'oil', 'vinegar', 'spice', 'rice', 'pasta', 'bread', 'sauce']
        protein_keywords = ['chicken', 'beef', 'pork', 'fish', 'turkey', 'lamb', 'tofu', 'beans', 'lentil']
        dairy_keywords = ['milk', 'cheese', 'butter', 'cream', 'yogurt', 'egg']
        
        for item in shopping_items:
            item_lower = item.lower()
            if any(keyword in item_lower for keyword in produce_keywords):
                produce_items.append(item)
            elif any(keyword in item_lower for keyword in protein_keywords):
                protein_items.append(item)
            elif any(keyword in item_lower for keyword in dairy_keywords):
                dairy_items.append(item)
            elif any(keyword in item_lower for keyword in pantry_keywords):
                pantry_items.append(item)
            else:
                other_items.append(item)
        
        # Display categorized items
        if produce_items:
            cart_recommendation += "\n**🥬 Fresh Produce:**\n"
            for i, item in enumerate(produce_items, 1):
                cart_recommendation += f"  {i}. {item}\n"
        
        if protein_items:
            cart_recommendation += "\n**🥩 Proteins:**\n"
            for i, item in enumerate(protein_items, 1):
                cart_recommendation += f"  {i}. {item}\n"
        
        if dairy_items:
            cart_recommendation += "\n**🥛 Dairy & Eggs:**\n"
            for i, item in enumerate(dairy_items, 1):
                cart_recommendation += f"  {i}. {item}\n"
        
        if pantry_items:
            cart_recommendation += "\n**🏺 Pantry Staples:**\n"
            for i, item in enumerate(pantry_items, 1):
                cart_recommendation += f"  {i}. {item}\n"
        
        if other_items:
            cart_recommendation += "\n**📦 Other Items:**\n"
            for i, item in enumerate(other_items, 1):
                cart_recommendation += f"  {i}. {item}\n"
        
        cart_recommendation += "\n"
    else:
        cart_recommendation += "**📋 Shopping List:**\n*No specific items identified from tool responses. Please refer to your recipe for ingredients.*\n\n"
    
    # Add store recommendations
    if store_recommendations:
        cart_recommendation += "**🏪 Recommended Stores:**\n"
        for i, store in enumerate(store_recommendations, 1):
            cart_recommendation += f"  {i}. {store}\n"
        cart_recommendation += "\n"
    
    # Add price information
    if price_comparisons:
        cart_recommendation += "**💰 Price Information:**\n"
        for i, price in enumerate(price_comparisons, 1):
            cart_recommendation += f"  {i}. {price}\n"
        cart_recommendation += "\n"
    
    # Add nutritional information if available
    if nutritional_info:
        cart_recommendation += "**🍎 Nutritional Notes:**\n"
        for info in nutritional_info[:3]:  # Limit to 3 entries
            cart_recommendation += f"• {info[:150]}...\n"
        cart_recommendation += "\n"
    
    # Add smart shopping tips based on the recipe
    cart_recommendation += """**💡 Smart Shopping Tips:**\n"""
    
    # Recipe-specific tips
    if 'chicken' in recipe_title.lower() or any('chicken' in item.lower() for item in shopping_items):
        cart_recommendation += "• Buy chicken in family packs and freeze portions for future meals\n"
    
    if any('vegetable' in item.lower() or 'produce' in item.lower() for item in shopping_items):
        cart_recommendation += "• Shop for produce early in the morning for the freshest selection\n"
    
    cart_recommendation += """• Compare unit prices rather than package prices for better value
• Check store apps for digital coupons before shopping
• Consider buying non-perishables in bulk if you cook this recipe often
• Don't shop when hungry to avoid impulse purchases
• Bring a reusable shopping list to stay organized

**🎯 Estimated Shopping Time:** 30-45 minutes
**💳 Money-Saving Tip:** Look for store brands on pantry staples - they're often 20-30% cheaper!

**Ready to go shopping? Happy cooking! 👨‍🍳👩‍🍳**"""
    
    display_messages.append(cart_recommendation)
    
    # Store comprehensive shopping cart data in state
    shopping_cart_data = {
        "items": {
            "produce": produce_items if 'produce_items' in locals() else [],
            "proteins": protein_items if 'protein_items' in locals() else [],
            "dairy": dairy_items if 'dairy_items' in locals() else [],
            "pantry": pantry_items if 'pantry_items' in locals() else [],
            "other": other_items if 'other_items' in locals() else [],
            "all_items": shopping_items
        },
        "stores": store_recommendations,
        "prices": price_comparisons,
        "nutritional_info": nutritional_info,
        "total_items": len(shopping_items),
        "recommendation_generated": True,
        "recipe_title": recipe_title
    }
    
    return {
        "display_messages": display_messages,
        "shopping_cart_data": shopping_cart_data,
        "workflow_stage": "shopping_cart_complete",
        "processing_complete": True
    }

