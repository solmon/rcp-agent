import asyncio
import uuid
import os
from langchain_core.messages import HumanMessage
from agent.graph import create_recipe_agent_with_mcp, recipe_agent
from agent.state import RecipeAgentState
from langfuse import Langfuse

# Langfuse monitoring setup
try:
    LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "dev-api-key")
    LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "dev-secret-key")
    LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    langfuse = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST
    )
    # langfuse = Langfuse(
    #     secret_key="sk-lf-38b56ec9-0194-4c38-9f3e-104cde792313",
    #     public_key="pk-lf-80b2ef91-821f-488a-9f1d-93ba683ada18",
    #     host="http://localhost:3000"
    # )
except ImportError:
    langfuse = None
    print("Langfuse not installed. Monitoring disabled.")


def display_messages(state_update):
    """Display messages from the state display_messages array."""
    if isinstance(state_update, dict):
        for node_name, node_output in state_update.items():
            if node_name != "__end__" and isinstance(node_output, dict):
                display_msgs = node_output.get("display_messages", [])
                for msg in display_msgs:
                    print(f"\n{msg}")


async def run_recipe_agent_stream(user_input: str):
    """Run the recipe agent with streaming output and recursive interrupt handling."""
    try:
        # Try to create agent with MCP tools
        agent = await create_recipe_agent_with_mcp()
    except Exception as e:
        print(f"❌ Error creating MCP agent: {e}")
        print("🔄 Using standard agent...")
        agent = recipe_agent
    
    # Create initial state
    initial_state = {
        "user_query": user_input,
        "messages": [HumanMessage(content=user_input)],
        "recipes": [],
        "display_messages": [],  # Initialize display messages
        "human_response": None,
        "needs_user_input": False,
        "recipe_confirmed": False,
        "recipe_plan_confirmed": False,
        "workflow_stage": "initial",
        "processing_complete": False,
        "tool_outputs": {},
        "pending_tool_calls": []  # Initialize pending tool calls
    }
    
    # Start the streaming execution with recursive interrupt handling
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}  # Generate unique thread ID for each conversation
    
    try:
        # Helper function to handle interrupts recursively
        async def handle_workflow_with_interrupts(input_state, config):
            """Recursively handle workflow execution with interrupts."""
            
            # Stream the agent execution
            async for chunk in agent.astream(input_state, config):
                # Print each node's output as it completes
                for node_name, node_output in chunk.items():
                    if node_name != "__end__":
                        print(f"📍 {node_name}: Processing...")
                        
                        # Handle different types of output
                        if isinstance(node_output, dict):
                            # Display messages from nodes
                            display_messages(chunk)
                            
                            # Handle workflow stage updates
                            if "workflow_stage" in node_output:
                                stage = node_output["workflow_stage"]
                                print(f"🔄 Workflow stage: {stage}")
                            
                            # Handle errors
                            if "error_message" in node_output and node_output["error_message"]:
                                print(f"❌ Error: {node_output['error_message']}")
                        
                        print()  # Add spacing between nodes
            
            # Check if we're interrupted and need user input
            current_state = agent.get_state(config)
            if current_state.next:
                print("⏸️  Waiting for user approval...")
                
                # Display any pending messages
                if hasattr(current_state, 'values') and current_state.values.get("display_messages"):
                    for msg in current_state.values["display_messages"]:
                        print(f"\n{msg}")
                
                # Get user response
                while True:
                    try:
                        user_response = input("\n🗣️  Your response: ").strip()
                        if user_response:
                            break
                        print("Please provide a response.")
                    except EOFError:
                        print("\n👋 Session ended.")
                        return None
                
                print(f"\n🔄 Continuing with your response: {user_response}")
                print("-" * 30)
                
                # Import Command for proper resumption
                from langgraph.types import Command
                
                # Resume with Command object instead of updating state manually
                resume_command = Command(resume=user_response, update={"human_response": user_response})
                
                # Recursive call to handle the next part of the workflow
                return await handle_workflow_with_interrupts(resume_command, config)
            
            else:
                # No more interrupts, workflow is complete
                print(f"✅ Workflow complete - no more interrupts")
                return current_state.values if hasattr(current_state, 'values') else None
        
        # Start the recursive workflow handling
        final_result = await handle_workflow_with_interrupts(initial_state, config)
        
        print("✅ Processing complete!")
        
        # Reset to initial state for next call
        return final_result
        
    except Exception as e:
        print(f"❌ Error during streaming: {e}")
        # Fallback to regular invoke
        try:
            result = await agent.ainvoke(initial_state, config)
            print("📋 Final result received")
            return result
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            return None


def main():
    print("🍳 Enhanced Recipe Agent CLI with Human-in-the-Loop")
    print("Type 'quit', 'exit', or 'q' to exit")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n🗣️  You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Happy cooking!")
                break
            
            if not user_input:
                continue
            
            print("🤔 Processing with streaming workflow...")
            print("-" * 30)
            
            # Run the streaming agent
            result = asyncio.run(run_recipe_agent_stream(user_input))
            
            print("-" * 50)
            
        except EOFError:
            # Handle EOF gracefully (e.g., when input is piped)
            print("\n👋 Happy cooking!")
            break
        except KeyboardInterrupt:
            print("\n👋 Happy cooking!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
