---
applyTo: '**'
---
Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.

---
applyTo: '**/apps/reciepe_agent/**'
---
Coding standards, domain knowledge, and preferences that AI should follow.

# Frameworks and tools
- Use the [LangGraph](https://langchain.com/docs/langgraph) framework for building agents.
- Use the [Model Context Protocol](https://modelcontextprotocol.org/) for agent communication.
- Use the [LangGraph CLI](https://langchain.com/docs/langgraph/cli) for running and testing agents.
- Use the [LangGraph Inspector](https://langchain.com/docs/langgraph/inspector) for debugging and inspecting agents.
- Use the [LangGraph Agentic Orchestrator](https://langchain.com/docs/langgraph/agentic-orchestrator) for managing agent interactions.
- Use the [LangGraph Agentic Recipe](https://langchain.com/docs/langgraph/agentic-recipe) for defining agent workflows.
- Use the [LangGraph Agentic Orchestrator CLI](https://langchain.com/docs/langgraph/agentic-orchestrator/cli) for running and testing agent workflows.
- Use the [LangGraph Agentic Recipe CLI](https://langchain.com/docs/langgraph/agentic-recipe/cli) for running and testing agent recipes.
- Use the [LangGraph Agentic Orchestrator Inspector](https://langchain.com/docs/langgraph/agentic-orchestrator/inspector) for debugging and inspecting agent workflows.
- Use the [LangGraph Agentic Recipe Inspector](https://langchain.com/docs/langgraph/agentic-recipe/inspector) for debugging and inspecting agent recipes.
- Use the [LangGraph Agentic Orchestrator Debugger](https://langchain.com/docs/langgraph/agentic-orchestrator/debugger) for debugging agent workflows.
- use langchain_mcp_adapters.client for MCP client interactions. (https://pypi.org/project/langchain-mcp-adapters/)

# Coding standards
- Follow LangGraph structure for the project

# Basic agent structure
- Use the `agent` directory for agent code.
- Use the `graph.py` file for defining the agent workflow.
- Use the `nodes.py` file for defining agent nodes.
- Use the `tools.py` file for defining agent tools.
- Use the `graph_interrupt.py` file for handling workflow interruptions.
- Use the `state.py` file for defining the agent state.
- Use the `config.py` file for agent configuration.
- Use the `main.py` file for running the agent.
- use the `langgraph` stream function to stream messages to the UI.
- Use the `langgraph` package for building agents.
- use the venv for running all the commands


