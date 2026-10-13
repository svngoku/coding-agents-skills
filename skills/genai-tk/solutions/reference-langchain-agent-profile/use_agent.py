"""Load the research profile from config/agents.yaml and run it."""

from genai_tk.agents.langchain import LangchainAgent

# The profile drives the agent; only the model is overridden here.
agent = LangchainAgent("research", llm="claude-haiku@openrouter")

result = agent.run("Summarise this week's open-weight model releases")
print(result)
