from langchain_core.prompts import HumanMessagePromptTemplate

# Chat prompt for recipe queries
RECIPE_CHAT_PROMPT = HumanMessagePromptTemplate.from_template(
    "User query: {user_query}\nContext: {context}\n\nProvide a helpful recipe-focused response."
)

# Chat prompt for grocery search
GROCERY_CHAT_PROMPT = HumanMessagePromptTemplate.from_template(
    "Please search for these ingredients in grocery stores: {ingredients}\nUser context: {user_query}\nAdditional context: {context}"
)

# Chat prompt for grocery search
GROCERY_EXEC_CHAT_PROMPT = HumanMessagePromptTemplate.from_template(
    "Recipe article content is: {user_query}\nContext: \n\nProvide a detailed plan for execution by agent and all the mcp tool calls necessary at once so that i can execute in parallel."    
)

RECIPE_ARTICLE_CHAT_PROMPT = HumanMessagePromptTemplate.from_template(
    "Recipe article content is: {selected_recipe}\nContext: {context}\n\nProvide a detailed plan for execution by agent."
)