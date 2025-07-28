# Recipe Agent

## Overview
The Recipe Agent is a LangGraph-based implementation designed to assist users in managing and executing recipe workflows. It leverages the LangGraph framework for building stateful workflows and integrates tools for enhanced functionality. The agent is capable of handling user queries, classifying intents, and executing tools to provide a seamless recipe management experience.

## Features
- **Intent Classification**: Classifies user queries into specific intents such as recipe search or general queries.
- **Recipe Confirmation**: Confirms recipes with the user before proceeding.
- **Tool Execution**: Executes tools for tasks like grocery list management.
- **MCP Integration**: Optionally integrates with Model Context Protocol (MCP) tools for extended functionality.
- **State Persistence**: Uses a memory-based checkpointer to persist state during workflow interruptions.

## Project Structure
```
rcp-agent/
├── agent/
│   ├── graph.py          # Defines the workflow and state graph
│   ├── nodes.py          # Contains logic for various nodes
│   ├── state.py          # Defines the agent state
│   ├── tools.py          # Defines tools used by the agent
├── prompts/
│   ├── chat_prompts.py   # Chat-specific prompts
│   ├── system_prompts.py # System-specific prompts
├── main.py               # Entry point for running the agent
├── pyproject.toml        # Project dependencies and configuration
├── README.md             # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone git@github.com:solmon/rcp-agent.git
   cd rcp-agent
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Running the Agent
To run the Recipe Agent, execute the following command:
```bash
python main.py
```

### Running with MCP Tools
To enable MCP tools, ensure the `langchain_mcp_adapters` package is installed and run:
```bash
python -c "from agent.graph import run_recipe_agent_with_mcp; import asyncio; asyncio.run(run_recipe_agent_with_mcp('Your query here'))"
```

## Development
### Adding New Nodes
1. Define the node logic in `nodes.py`.
2. Add the node to the workflow in `graph.py`.

### Adding New Tools
1. Define the tool in `tools.py`.
2. Register the tool in the `recipe_tools` list.

## Testing
Run the following command to execute tests:
```bash
pytest
```

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- [LangGraph](https://langchain.com/docs/langgraph) for the workflow framework.
- [Model Context Protocol](https://modelcontextprotocol.org/) for agent communication.
- [LangChain MCP Adapters](https://pypi.org/project/langchain-mcp-adapters/) for MCP client interactions.
