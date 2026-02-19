# Coding Agents Skills

> A curated collection of specialized skills for AI coding agents, designed to enhance agent capabilities across software development, architecture, and framework integration.

## 📖 Overview

This repository provides production-ready skill modules for AI coding agents, enabling them to:

- Apply software design patterns and architectural principles
- Work with modern AI frameworks and tools
- Follow best practices in code generation and refactoring
- Integrate seamlessly with development workflows

Each skill is self-contained with comprehensive documentation, reference materials, and practical examples.

## 🗂️ Project Structure

```
coding-agents-skills/
├── skills/
│   ├── ddd/                          # Domain-Driven Design
│   │   ├── SKILL.md                  # Core skill definition
│   │   └── references/               # Reference documentation
│   │       ├── strategic-design.md
│   │       ├── tactical-design.md
│   │       ├── architecture-patterns.md
│   │       ├── event-storming.md
│   │       ├── python-patterns.md
│   │       ├── typescript-patterns.md
│   │       └── code-review.md
│   │
│   ├── langchain/                    # LangChain framework
│   │   ├── SKILL.md
│   │   └── references/
│   │
│   └── smolagents/                   # Hugging Face SmolAgents
│       ├── SKILL.md
│       └── references/
│
└── README.md
```

## 🎯 Available Skills

### 1. Domain-Driven Design (DDD)

**Status:** ✅ Complete

**Description:** Comprehensive DDD skill for building software that reflects deep understanding of business domains.

**Capabilities:**

- Strategic design (bounded contexts, subdomains, context maps)
- Tactical design (entities, value objects, aggregates, repositories)
- Architecture patterns (Hexagonal, CQRS, Event Sourcing, Clean Architecture)
- Event Storming facilitation
- Language-specific implementations (Python with Pydantic/FastAPI, TypeScript with NestJS)
- DDD code review guidance

**Use Cases:**

- Designing new systems with DDD principles
- Refactoring existing codebases toward DDD
- Generating code scaffolding (entities, aggregates, repositories)
- Performing code reviews with a DDD lens

**Reference Files:** 7 comprehensive guides

---

### 2. LangChain

**Status:** ✅ Complete

**Description:** Build AI applications with LangChain framework.

**Capabilities:**

- Chain construction and composition
- Memory management
- Agent creation and orchestration
- Tool integration
- RAG (Retrieval-Augmented Generation) patterns

---

### 3. SmolAgents

**Status:** ✅ Complete

**Description:** Build AI agents with Hugging Face's minimalist SmolAgents framework.

**Capabilities:**

- CodeAgent and ToolCallingAgent creation
- Custom tool development
- MCP (Model Context Protocol) integration
- Multi-agent systems
- Secure code execution (E2B, Docker, Blaxel)
- Model configuration (HF Inference, LiteLLM, Transformers, Ollama)
- Agentic RAG and text-to-SQL pipelines
- Web browsing agents

---

## 🚀 Usage

### For AI Coding Agents

Each skill can be loaded by agents to enhance their capabilities:

1. **Load a skill:** Reference the `SKILL.md` file in the appropriate skill directory
2. **Access references:** Each skill includes detailed reference documentation in its `references/` folder
3. **Apply patterns:** Follow the workflows and examples provided in the skill documentation

### For Developers

**Clone the repository:**

```bash
git clone https://github.com/svngoku/coding-agents-skills.git
cd coding-agents-skills
```

**Browse skills:**

```bash
# View available skills
ls skills/

# Read a specific skill
cat skills/ddd/SKILL.md

# Explore reference materials
ls skills/ddd/references/
```

### Integration Examples

#### With SmolAgents

```python
from smolagents import CodeAgent, HfApiModel

# Load DDD skill for architectural guidance
agent = CodeAgent(
    tools=[],
    model=HfApiModel(),
    additional_authorized_imports=["pydantic", "typing"]
)

result = agent.run(
    "Design a bounded context for an e-commerce order management system "
    "following DDD principles. Use the DDD skill reference."
)
```

#### With LangChain

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI

# Agent with skill context
llm = ChatOpenAI(temperature=0)
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

result = agent_executor.invoke({
    "input": "Review this code for DDD compliance",
    "skill_context": "skills/ddd/SKILL.md"
})
```

## 📋 Skill Template

Each skill follows a consistent structure:

### SKILL.md Format

```markdown
# Skill Name

| name       | description                                            |
| ---------- | ------------------------------------------------------ |
| skill-name | Detailed description of when and how to use this skill |

## Overview

[Brief introduction]

## Quick Reference

[Table linking to reference files]

## Core Workflow

[Step-by-step usage guide]

## Implementation Guidelines

[Concrete examples and patterns]

## Anti-Patterns to Avoid

[Common mistakes]

## When to Use / Not Use

[Decision criteria]
```

### References Structure

- Each skill has a `references/` directory
- Reference files are in Markdown format
- Cover specific aspects of the skill in depth
- Include code examples in relevant languages

## 🤝 Contributing

Contributions are welcome! To add a new skill:

1. **Fork the repository**
2. **Create a new skill directory** under `skills/`
3. **Follow the skill template:**
   - Create `SKILL.md` with the standard structure
   - Add `references/` directory with detailed documentation
   - Include practical examples and code samples
4. **Submit a pull request**

### Skill Guidelines

- Skills should be atomic and focused on a single domain
- Include both conceptual explanations and practical examples
- Provide language-specific implementations where relevant
- Document anti-patterns and common mistakes
- Keep reference files modular and cross-referenced

## 📚 Roadmap

- [x] Domain-Driven Design skill
- [x] SmolAgents skill
- [ ] LangChain skill (in progress)
- [ ] Testing patterns skill
- [ ] API design skill
- [ ] Database design skill
- [ ] Microservices patterns skill
- [ ] Security best practices skill
- [ ] Performance optimization skill

## 🔗 Related Projects

- [Hugging Face SmolAgents](https://github.com/huggingface/smolagents) - Minimalist AI agent framework
- [LangChain](https://github.com/langchain-ai/langchain) - Building applications with LLMs
- [Model Context Protocol](https://modelcontextprotocol.io/) - Standard for connecting AI systems

## 📄 License

MIT License - see LICENSE file for details

## 👤 Author

**svngoku**

- GitHub: [@svngoku](https://github.com/svngoku)

## 🙏 Acknowledgments

- Inspired by the need for reusable, production-ready AI agent skills
- Built with insights from software architecture patterns and modern AI frameworks
- Community feedback and contributions welcome

---

**Note:** This is an evolving collection. Skills are added and updated based on practical needs in AI-assisted software development.
