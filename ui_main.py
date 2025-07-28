"""
Simple Web UI for Recipe Agent
A clean web interface for the recipe agent with real-time interaction.
"""

import asyncio
import uuid
import json
import logging
from typing import Dict, Any, AsyncGenerator, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import create_recipe_agent_with_mcp, recipe_agent
from agent.state import RecipeAgentState

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global agent instance and session store
global_agent = None
active_sessions = {}  # Store active conversation sessions


async def init_agent():
    """Initialize the recipe agent."""
    global global_agent
    try:
        # Try to create agent with MCP tools
        global_agent = await create_recipe_agent_with_mcp()
        logger.info("✅ Recipe agent with MCP tools initialized successfully")
    except Exception as e:
        logger.warning(f"❌ Error creating MCP agent: {e}")
        logger.info("🔄 Using standard agent...")
        global_agent = recipe_agent
    return global_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🚀 Starting Recipe Agent UI...")
    await init_agent()
    yield
    # Shutdown
    logger.info("👋 Shutting down Recipe Agent UI...")


# Create FastAPI app
app = FastAPI(
    title="Recipe Agent UI",
    description="A beautiful interface for the Recipe Agent",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


async def process_query_stream(user_message: str, session_id: str = None) -> AsyncGenerator[str, None]:
    """
    Process user query and stream responses using Server-Sent Events with session persistence.
    """
    try:
        # Generate or use existing session ID
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Create config with persistent thread ID
        config = {
            "configurable": {
                "thread_id": session_id
            }
        }
        
        logger.info(f"🤔 Processing query: {user_message} (Session: {session_id})")
        
        # Check if this is a resume from an interrupt
        is_resume = session_id in active_sessions and active_sessions[session_id].get("interrupted", False)
        
        if is_resume:
            # This is a resume from an interrupt
            yield f"data: {json.dumps({'type': 'status', 'message': 'Resuming from where we left off...', 'stage': 'resuming', 'session_id': session_id})}\n\n"
            
            # Import Command for proper resumption
            from langgraph.types import Command
            
            # Resume with Command object
            resume_command = Command(resume=user_message, update={"human_response": user_message})
            
            # Mark as no longer interrupted
            active_sessions[session_id]["interrupted"] = False
            
            # Continue the workflow
            input_state = resume_command
            
        else:
            # This is a new conversation or continuation
            # Create initial agent state
            initial_state = {
                "user_query": user_message,
                "messages": [HumanMessage(content=user_message)],
                "recipes": [],
                "display_messages": [],
                "human_response": None,
                "needs_user_input": False,
                "recipe_confirmed": False,
                "recipe_plan_confirmed": False,
                "workflow_stage": "initial",
                "processing_complete": False,
                "tool_outputs": {},
                "pending_tool_calls": []
            }
            
            # Store session info
            active_sessions[session_id] = {
                "thread_id": session_id,
                "interrupted": False,
                "last_message": user_message
            }
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing your request...', 'stage': 'starting', 'session_id': session_id})}\n\n"
            
            input_state = initial_state
        
        # Stream the agent execution
        async for chunk in global_agent.astream(input_state, config):
            for node_name, node_output in chunk.items():
                if node_name != "__end__" and isinstance(node_output, dict):
                    
                    # Handle display messages
                    display_msgs = node_output.get("display_messages", [])
                    for msg in display_msgs:
                        
                        # Send message chunk
                        yield f"data: {json.dumps({'type': 'message', 'content': msg, 'node': node_name, 'session_id': session_id})}\n\n"

                    # Handle workflow stage updates
                    if "workflow_stage" in node_output:
                        stage = node_output["workflow_stage"]
                        stage_msg = f"🔄 Workflow stage: {stage}"
                        logger.info(stage_msg)
                        
                        # Send stage update
                        yield f"data: {json.dumps({'type': 'stage', 'stage': stage, 'message': stage_msg, 'session_id': session_id})}\n\n"

                    # Handle errors
                    if "error_message" in node_output and node_output["error_message"]:
                        error_msg = f"❌ Error: {node_output['error_message']}"
                        logger.error(error_msg)
                        yield f"data: {json.dumps({'type': 'error', 'message': error_msg, 'session_id': session_id})}\n\n"
        
        # Check if we're interrupted and need user input
        current_state = global_agent.get_state(config)
        if current_state.next:
            # We're interrupted - mark session as interrupted
            active_sessions[session_id]["interrupted"] = True
            
            # Display any pending messages
            if hasattr(current_state, 'values') and current_state.values.get("display_messages"):
                for msg in current_state.values["display_messages"]:
                    yield f"data: {json.dumps({'type': 'message', 'content': msg, 'session_id': session_id})}\n\n"
            
            # Send interrupt signal to UI
            yield f"data: {json.dumps({'type': 'interrupt', 'message': '⏸️ Waiting for your approval...', 'session_id': session_id, 'requires_input': True})}\n\n"
            
        else:
            # No more interrupts, workflow is complete
            logger.info(f"✅ Workflow complete - no more interrupts for session {session_id}")
            
            # Send completion
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Workflow completed!', 'session_id': session_id})}\n\n"
            
            # Clean up session
            if session_id in active_sessions:
                del active_sessions[session_id]
        
        logger.info("✅ Query processing complete")

    except Exception as e:
        error_msg = f"❌ Error processing request: {str(e)}"
        logger.error(error_msg)
        yield f"data: {json.dumps({'type': 'error', 'message': error_msg, 'session_id': session_id if 'session_id' in locals() else 'unknown'})}\n\n"


async def handle_workflow_with_interrupts(input_state, config, session_id):
    """Recursively handle workflow execution with interrupts and streaming."""
    
    # Stream the agent execution
    response_parts = []
    
    async def stream_generator():
        async for chunk in global_agent.astream(input_state, config):
            for node_name, node_output in chunk.items():
                if node_name != "__end__" and isinstance(node_output, dict):
                    
                    # Handle display messages
                    display_msgs = node_output.get("display_messages", [])
                    for msg in display_msgs:
                        response_parts.append(msg)
                        
                        # Send message chunk
                        yield f"data: {json.dumps({'type': 'message', 'content': msg, 'node': node_name, 'session_id': session_id})}\n\n"

                    # Handle workflow stage updates
                    if "workflow_stage" in node_output:
                        stage = node_output["workflow_stage"]
                        stage_msg = f"🔄 Workflow stage: {stage}"
                        logger.info(stage_msg)
                        
                        # Send stage update
                        yield f"data: {json.dumps({'type': 'stage', 'stage': stage, 'message': stage_msg, 'session_id': session_id})}\n\n"

                    # Handle errors
                    if "error_message" in node_output and node_output["error_message"]:
                        error_msg = f"❌ Error: {node_output['error_message']}"
                        logger.error(error_msg)
                        yield f"data: {json.dumps({'type': 'error', 'message': error_msg, 'session_id': session_id})}\n\n"
        
        # Check if we're interrupted and need user input
        current_state = global_agent.get_state(config)
        if current_state.next:
            # We're interrupted - mark session as interrupted
            active_sessions[session_id]["interrupted"] = True
            
            # Display any pending messages
            if hasattr(current_state, 'values') and current_state.values.get("display_messages"):
                for msg in current_state.values["display_messages"]:
                    yield f"data: {json.dumps({'type': 'message', 'content': msg, 'session_id': session_id})}\n\n"
            
            # Send interrupt signal to UI
            yield f"data: {json.dumps({'type': 'interrupt', 'message': '⏸️ Waiting for your approval...', 'session_id': session_id, 'requires_input': True})}\n\n"
            
        else:
            # No more interrupts, workflow is complete
            logger.info(f"✅ Workflow complete - no more interrupts for session {session_id}")
            
            # Clean up session
            if session_id in active_sessions:
                del active_sessions[session_id]
    
    return stream_generator()


@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    """Stream chat responses using Server-Sent Events."""
    try:
        body = await request.json()
        user_message = body.get("message", "")
        session_id = body.get("session_id", None)  # Get session ID from request
        
        if not user_message:
            return {"error": "No message provided"}
        
        return StreamingResponse(
            process_query_stream(user_message, session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )
    except Exception as e:
        logger.error(f"Error in chat stream: {e}")
        return {"error": str(e)}





@app.get("/alt", response_class=HTMLResponse)
async def serve_static_html():
    """Serve the static HTML file."""
    with open("static/index.html", "r") as f:
        return f.read()


@app.get("/simple", response_class=HTMLResponse)
async def simple_test():
    """Serve a simple test page to debug JavaScript issues."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple Test</title>
    </head>
    <body>
        <h1>Simple JavaScript Test</h1>
        <button onclick="testFunction()">Test Button</button>
        <input id="testInput" type="text" placeholder="Type here...">
        <button onclick="sendTestMessage()">Send Test</button>
        <div id="output"></div>
        
        <script>
            function testFunction() {
                console.log('Test button clicked!');
                document.getElementById('output').innerHTML = 'Button works!';
            }
            
            function sendTestMessage() {
                const input = document.getElementById('testInput');
                const output = document.getElementById('output');
                output.innerHTML = 'You typed: ' + input.value;
                console.log('Send test message called with:', input.value);
            }
            
            console.log('Simple test page loaded');
        </script>
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI page."""
    with open("static/index.html", "r") as f:
        return f.read()


@app.get("/test")
async def test_endpoint():
    """Simple test endpoint."""
    return {"message": "Backend is working!", "status": "ok"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Recipe Agent UI is running",
        "agent_initialized": global_agent is not None
    }


@app.get("/api/agent/status")
async def agent_status():
    """Get agent status."""
    return {
        "agent_available": global_agent is not None,
        "agent_type": "MCP-enabled" if global_agent else "Standard",
        "version": "1.0.0",
        "active_sessions": len(active_sessions)
    }


@app.get("/api/sessions")
async def get_sessions():
    """Get active sessions."""
    return {
        "active_sessions": list(active_sessions.keys()),
        "session_count": len(active_sessions)
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session."""
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"message": f"Session {session_id} deleted"}
    return {"message": f"Session {session_id} not found"}



    """Main entry point for the UI server."""
    print("🍳 Starting Recipe Agent UI Server...")
    print("🌐 The UI will be available at: http://localhost:8001")
    print("📋 Health check available at: http://localhost:8001/health")
    print("🔧 Agent status available at: http://localhost:8001/api/agent/status")
    print("-" * 60)
    
    # Run the server
    uvicorn.run(
        "ui_main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )


def main():
    """Main entry point for the UI server."""
    print("🍳 Starting Recipe Agent UI Server...")
    print("🌐 The UI will be available at: http://localhost:8001")
    print("📋 Health check available at: http://localhost:8001/health")
    print("🔧 Agent status available at: http://localhost:8001/api/agent/status")
    print("-" * 60)
    
    # Run the server
    uvicorn.run(
        "ui_main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )


if __name__ == "__main__":
    main()
