"""Load the Research profile from config/agents/langchain.yaml and run it."""
from genai_tk.agents.langchain import LangchainAgent

# Profile "Research" comes from config/agents/langchain.yaml.
# Overrides demonstrate the model_id@provider format and the Docker sandbox.
agent = LangchainAgent(
    "Research",
    llm="gpt_4o@openai",
    sandbox="docker",
)

result = agent.run("Summarise recent AI news")
print(result)
