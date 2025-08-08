import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from openai import OpenAI

class ReasoningChatOpenAI(ChatOpenAI):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Convert LangChain messages to OpenAI format with proper role handling
        openai_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                if hasattr(msg, 'role'):
                    role = msg.role
                elif msg.__class__.__name__ == 'HumanMessage':
                    role = "user"
                elif msg.__class__.__name__ == 'AIMessage':
                    role = "assistant"
                elif msg.__class__.__name__ == 'SystemMessage':
                    role = "system"
                else:
                    role = "user"
                
                openai_messages.append({
                    "role": role,
                    "content": msg.content
                })
        
        # Handle vLLM limitations - DeepSeek models don't support tools
        filtered_kwargs = kwargs.copy()
        
        # For DeepSeek models, completely remove tool-related parameters
        if "deepseek" in self.model_name.lower():
            filtered_kwargs = {k: v for k, v in filtered_kwargs.items() 
                             if k not in ['tools', 'tool_choice']}
        else:
            # For other models, just remove problematic "auto" tool_choice
            if 'tool_choice' in filtered_kwargs and filtered_kwargs['tool_choice'] == "auto":
                filtered_kwargs.pop('tool_choice')
        
        # Use OpenAI client directly
        client = OpenAI(api_key=self.openai_api_key, base_url=self.openai_api_base)
        
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                temperature=self.temperature,
                **filtered_kwargs
            )
        except Exception as e:
            # If there's a tool-related error, fallback to no tools
            if "tool" in str(e).lower():
                fallback_kwargs = {k: v for k, v in filtered_kwargs.items() 
                                 if k not in ['tools', 'tool_choice']}
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=openai_messages,
                    temperature=self.temperature,
                    **fallback_kwargs
                )
            else:
                raise e
        
        # Extract all relevant content from OpenAI response
        message = response.choices[0].message
        reasoning_content = getattr(message, 'reasoning_content', None)
        tool_calls = getattr(message, 'tool_calls', None)
        
        # Store all metadata in additional_kwargs
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.messages import AIMessage
        import json
        
        additional_kwargs = {}
        if reasoning_content:
            additional_kwargs['reasoning_content'] = reasoning_content
        if tool_calls:
            additional_kwargs['tool_calls'] = [
                {
                    'id': tc.id,
                    'type': tc.type,
                    'function': {
                        'name': tc.function.name,
                        'arguments': tc.function.arguments
                    }
                } for tc in tool_calls
            ]
        
        # Add usage information if available
        if hasattr(response, 'usage') and response.usage:
            additional_kwargs['usage'] = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        
        # Add model and other response metadata
        additional_kwargs['model'] = response.model
        additional_kwargs['finish_reason'] = response.choices[0].finish_reason
        
        # Parse tool calls for LangChain format
        langchain_tool_calls = None
        if tool_calls:
            langchain_tool_calls = []
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                
                langchain_tool_calls.append({
                    'name': tc.function.name,
                    'args': args,
                    'id': tc.id
                })
        
        ai_message = AIMessage(
            content=message.content or "",
            additional_kwargs=additional_kwargs
        )
        
        # Only add tool_calls if they exist
        if langchain_tool_calls:
            ai_message.tool_calls = langchain_tool_calls
        
        generation = ChatGeneration(
            message=ai_message,
            generation_info={
                'finish_reason': response.choices[0].finish_reason,
                'model': response.model,
                'usage': additional_kwargs.get('usage', {})
            }
        )
        return ChatResult(generations=[generation])


def get_default_model(mode=None):
    """Get the default model for the agent."""
    model_type = os.getenv("MODEL_TYPE", "gemini").lower()
    if model_type == "gemini" or mode == "recipe":
        return get_gemini_model()
    elif model_type == "anthropic":
        return get_anthropic_model()
    elif model_type == "qwen":
        return get_qwen_model()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")    

def get_gemini_model():
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
        transport="rest",
        client_options={
            "api_endpoint": "https://generativelanguage.googleapis.com"
        },
        model_kwargs={
            "enable_thinking": True  # If you want to enable this feature,            
        }
    )
    """Get the Gemini model for the agent."""
    return llm

def get_anthropic_model():
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model_name="claude-sonnet-4-20250514", max_tokens=64000)

def get_qwen_model():
    llm_with_reasoning = ReasoningChatOpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        temperature=0
    )
    return llm_with_reasoning
