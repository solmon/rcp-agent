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
        "workflow_stage": "initial"
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
        
        recipe_plan = state.get("plan_extract", {})
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
                    "workflow_stage": "recipe_display"
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
                "recipe_plan_confirmed": False                
            }
        elif mode == "execute":
            messages.append(AIMessage(content=response.content))
            
            # Check if there are tool calls to execute
            if hasattr(response, 'tool_calls') and response.tool_calls:
                return {
                    "messages": messages,
                    "workflow_stage": "tool_execution",
                    "pending_tool_calls": response.tool_calls,
                    "tool_outputs": {}
                }
            else:
                # No tool calls, return response directly
                return {
                    "messages": messages,
                    "workflow_stage": "recipe_execution_complete",
                    "tool_outputs": {"response": response.content}
                }
        else:
            messages.append(AIMessage(content=response.content))
            recipes = _extract_recipes_from_response(response)
            if recipes:
                state["recipes"] = recipes
            return {
                "messages": messages,
                "recipes": state["recipes"],
                "workflow_stage": "recipe_display"                
            }

    except Exception as e:
        return {
            "error_message": f"Error in LLM node: {str(e)}",
            "processing_complete": True
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
                "workflow_stage": "recipe_planning",
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
    
    for tool_id, output in tool_outputs.items():
        if output.get("status") == "success":
            tool_name = output.get("tool_name", "")
            result = output.get("result", "")
            
            # Extract shopping information based on tool type
            if "search" in tool_name.lower() or "find" in tool_name.lower():
                # Parse search results for items and stores
                if isinstance(result, dict):
                    items = result.get("items", [])
                    stores = result.get("stores", [])
                    prices = result.get("prices", [])
                    
                    shopping_items.extend(items)
                    store_recommendations.extend(stores)
                    price_comparisons.extend(prices)
                elif isinstance(result, str):
                    # Parse text results for shopping information
                    lines = result.split('\n')
                    for line in lines:
                        if any(keyword in line.lower() for keyword in ['store', 'shop', 'market']):
                            store_recommendations.append(line.strip())
                        elif any(keyword in line.lower() for keyword in ['item', 'ingredient', 'product']):
                            shopping_items.append(line.strip())
                        elif any(keyword in line.lower() for keyword in ['price', '$', 'cost']):
                            price_comparisons.append(line.strip())
            
            elif "price" in tool_name.lower() or "compare" in tool_name.lower():
                # Handle price comparison results
                if isinstance(result, (list, dict)):
                    price_comparisons.append(str(result))
                else:
                    price_comparisons.append(result)
    
    # Generate shopping cart recommendation
    recipe_title = selected_recipe.get('title', 'Your Recipe')
    
    cart_recommendation = f"""🛒 **Shopping Cart Recommendation for {recipe_title}**

"""
    
    if shopping_items:
        cart_recommendation += "**📋 Recommended Items:**\n"
        unique_items = list(set(shopping_items))[:10]  # Limit to 10 unique items
        for i, item in enumerate(unique_items, 1):
            cart_recommendation += f"{i}. {item}\n"
        cart_recommendation += "\n"
    
    if store_recommendations:
        cart_recommendation += "**🏪 Recommended Stores:**\n"
        unique_stores = list(set(store_recommendations))[:5]  # Limit to 5 unique stores
        for i, store in enumerate(unique_stores, 1):
            cart_recommendation += f"{i}. {store}\n"
        cart_recommendation += "\n"
    
    if price_comparisons:
        cart_recommendation += "**💰 Price Information:**\n"
        unique_prices = list(set(price_comparisons))[:5]  # Limit to 5 price entries
        for i, price in enumerate(unique_prices, 1):
            cart_recommendation += f"{i}. {price}\n"
        cart_recommendation += "\n"
    
    # Add shopping tips
    cart_recommendation += """**💡 Shopping Tips:**
• Compare prices across different stores before purchasing
• Check for seasonal discounts and bulk buying options
• Consider organic vs. regular options based on your preference
• Don't forget to check store loyalty programs for additional savings

**Ready to go shopping? Happy cooking! 👨‍🍳👩‍🍳**"""
    
    display_messages.append(cart_recommendation)
    
    # Store shopping cart data in state
    shopping_cart_data = {
        "items": shopping_items,
        "stores": store_recommendations,
        "prices": price_comparisons,
        "recommendation_generated": True
    }
    
    return {
        "display_messages": display_messages,
        "shopping_cart_data": shopping_cart_data,
        "workflow_stage": "shopping_cart_complete",
        "processing_complete": True
    }

