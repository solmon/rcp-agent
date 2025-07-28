# Recipe Agent CopilotKit UI

This is a beautiful web interface for the Recipe Agent built with CopilotKit and FastAPI.

## Features

- 🍳 Interactive chat interface for recipe queries
- 🎨 Beautiful, responsive design
- ⚡ Real-time streaming responses
- 🔧 Built-in health checks and status monitoring
- 🌐 CORS-enabled for development

## Quick Start

### 1. Install Dependencies

```bash
# Install the new dependencies
uv sync
```

### 2. Start the UI Server

```bash
# Option 1: Using the task runner
uv run poe ui

# Option 2: Direct command
uv run python ui_main.py
```

### 3. Access the UI

Open your browser and go to:
- **Main UI**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **Agent Status**: http://localhost:8000/api/agent/status

## Usage

1. The UI loads with a welcoming interface
2. Start chatting with the recipe agent in the chat window
3. Ask questions like:
   - "Find me a recipe for chocolate cake"
   - "What ingredients do I need for pasta carbonara?"
   - "Create a meal plan for this week"
   - "Help me make a shopping list"

## Architecture

- **FastAPI**: Backend web framework
- **CopilotKit**: AI chat interface framework
- **LangGraph**: Agent workflow orchestration
- **React**: Frontend UI components (loaded via CDN)

## Development

The UI server runs with auto-reload enabled by default. Any changes to `ui_main.py` will automatically restart the server.

### Key Components

1. **Agent Integration**: Seamlessly integrates with your existing LangGraph recipe agent
2. **Streaming Support**: Real-time message streaming for better user experience
3. **Error Handling**: Graceful error handling and user feedback
4. **Session Management**: Simple single-session approach as requested

## Original CLI

The original CLI interface remains unchanged in `main.py` and can still be used:

```bash
# Run the CLI version
uv run python main.py
```

## Troubleshooting

1. **Port already in use**: Change the port in `ui_main.py` (line with `port=8000`)
2. **Agent not loading**: Check the health endpoint at `/health`
3. **Dependencies missing**: Run `uv sync` to install all dependencies

Enjoy cooking with your AI assistant! 🍳✨
