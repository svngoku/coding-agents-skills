# Coding Agents Skills

> A curated collection of specialized skills for AI coding agents, designed to enhance agent capabilities across software development, architecture, deployment, and framework integration.

## 📖 Overview

This repository provides production-ready skill modules for AI coding agents, enabling them to:
- Apply software design patterns and architectural principles
- Work with modern AI frameworks and tools (LangChain, SmolAgents, genai-tk)
- Deploy and operate applications on cloud platforms (Scalingo)
- Follow best practices in code generation, refactoring, and UI development
- Integrate seamlessly with development workflows

Each skill is self-contained with comprehensive documentation, reference materials, and practical examples.

## 🗂️ Project Structure

```
coding-agents-skills/
├── skills/
│   ├── adaption-ai/                    # Adaption AI SDK for synthetic data augmentation
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── api-reference.md
│   │       └── guides.md
│   │
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
│   ├── genai-tk-skill/               # GenAI Toolkit (YAML-driven agent framework)
│   │   ├── SKILL.md
│   │   ├── AGENTS.md
│   │   └── references/
│   │       ├── agents.md
│   │       ├── baml-structured.md
│   │       ├── cli-and-init.md
│   │       ├── configuration.md
│   │       └── rag.md
│   │
│   ├── langchain/                    # LangChain framework
│   │   ├── SKILL.md
│   │   └── references/
│   │
│   ├── scalingo/                      # Scalingo European PaaS deployment
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── addons-databases.md
│   │       ├── buildpacks.md
│   │       ├── cli-reference.md
│   │       ├── deployment.md
│   │       ├── manifest-review-apps.md
│   │       ├── scaling-autoscaler.md
│   │       └── terraform-iac.md
│   │
│   ├── smolagents/                   # Hugging Face SmolAgents
│   │   ├── SKILL.md
│   │   └── references/
│   │
│   ├── ui/                           # UI/UX best practices for agent-built interfaces
│   │   └── SKILL.md
│   │
│   └── unsloth-hf-jobs/              # Unsloth fine-tuning on Hugging Face Jobs
│       ├── SKILL.md
│       └── scripts/
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

**Description:** Build AI applications with the LangChain framework.

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

### 4. Adaption AI SDK
**Status:** ✅ Complete

**Description:** Build dataset augmentation pipelines with Adaption's Adaptive Data platform for synthetic data generation and fine-tuning preparation.

**Capabilities:**
- Upload and import datasets (local files, Hugging Face, Kaggle)
- Run augmentation/adaptation jobs with brand controls and recipe specifications
- Hallucination mitigation via web-search grounding
- DPO preference pair generation and deduplication
- Quality evaluation and export via presigned URLs
- Async client support with exponential backoff polling

**Use Cases:**
- Synthetic data generation for fine-tuning
- Dataset augmentation pipelines
- Grounding-based hallucination reduction on training data

---

### 5. genai-tk — GenAI & Agentic Toolkit
**Status:** ✅ Complete

**Description:** YAML-driven wrapper over LangChain, LangGraph, and 100+ LLM providers. Inversion-of-control layer where profiles in YAML drive factories that produce LangChain runtime objects.

**Capabilities:**
- `model_id@provider` LLM and embeddings factories
- Four bundled agent frameworks (ReAct, Deep, Deer-flow, SmolAgents)
- Unified `LangchainAgent` entry point with profile-based configuration
- `RetrieverFactory` with six retriever types (vector, BM25, ensemble, reranked, pg_hybrid, zero_entropy)
- BAML structured extraction
- OpenSandbox Docker integration for secure code execution
- MCP server registry and SkillsMiddleware for on-demand domain knowledge
- CLI scaffolding with `cli init`

**Use Cases:**
- Building production-grade agent systems with YAML configuration
- Multi-step planning with Deep agents and sandboxed execution
- Deep web research with Deer-flow
- Code-first automation with SmolAgents

---

### 6. Scalingo
**Status:** ✅ Complete

**Description:** Deploy and operate web applications on Scalingo, a European (French) Platform-as-a-Service with Heroku-compatible buildpacks and sovereign cloud regions.

**Capabilities:**
- App creation, deployment, and scaling via CLI and git
- Managed database addons (PostgreSQL, MySQL, MongoDB, Redis, OpenSearch, InfluxDB)
- Horizontal and vertical container scaling with autoscaler support
- `scalingo.json` manifest and review app configuration
- Terraform Infrastructure-as-Code provider
- SecNumCloud-qualified region (`osc-secnum-fr1`) for French public-sector workloads
- Migration guidance from Heroku

**Use Cases:**
- Deploying web apps to a European sovereign PaaS
- Managing production databases and background workers
- Automating infrastructure with Terraform
- Meeting French public-sector compliance requirements (HDS, SecNumCloud)

---

### 7. UI Skills
**Status:** ✅ Complete

**Description:** Opinionated constraints for building better interfaces with agents. Ensures accessibility, performance, and consistent design quality in AI-generated UI code.

**Capabilities:**
- Tailwind CSS and motion/react animation guidelines
- Accessible component primitives (Base UI, React Aria, Radix)
- Interaction best practices (AlertDialog for destructive actions, structural skeletons for loading)
- Animation constraints (compositor-only props, 200ms limit, reduced-motion support)
- Typography and layout rules (text-balance, tabular-nums, z-index scale)
- Performance guidelines (no large blur, no will-change outside animations)

**Use Cases:**
- Reviewing agent-generated UI for quality and accessibility
- Ensuring consistent Tailwind CSS usage
- Preventing common AI-generated UI anti-patterns

---

### 8. Unsloth Training on HF Jobs
**Status:** ✅ Complete

**Description:** Fine-tune LLMs and VLMs using Unsloth on Hugging Face on-demand cloud GPUs with UV scripts.

**Capabilities:**
- VLM fine-tuning (Qwen3-VL, Gemma 3) with image + message datasets
- Continued pretraining and domain adaptation
- LoRA fine-tuning with configurable rank and learning rate
- Trackio monitoring integration
- Automated dependency management via UV scripts

**Use Cases:**
- Fine-tuning vision-language models on custom datasets
- Domain adaptation with continued pretraining
- Running GPU training jobs without local hardware

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
---
name: skill-name
description: Detailed description of when and how to use this skill
---

# Skill Name

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
- Each skill has a `references/` directory (when applicable)
- Reference files are in Markdown format
- Cover specific aspects of the skill in depth
- Include code examples in relevant languages

## 🤝 Contributing

Contributions are welcome! To add a new skill:

1. **Fork the repository**
2. **Create a new skill directory** under `skills/`
3. **Follow the skill template:**
   - Create `SKILL.md` with the standard structure (including YAML frontmatter with `name` and `description`)
   - Add `references/` directory with detailed documentation when needed
   - Include practical examples and code samples
4. **Submit a pull request**

### Skill Guidelines
- Skills should be atomic and focused on a single domain
- Include both conceptual explanations and practical examples
- Provide language-specific implementations where relevant
- Document anti-patterns and common mistakes
- Keep reference files modular and cross-referenced
- Use YAML frontmatter with `name` and `description` for machine readability

## 📚 Roadmap

- [x] Domain-Driven Design skill
- [x] SmolAgents skill
- [x] LangChain skill
- [x] Adaption AI SDK skill
- [x] genai-tk skill
- [x] Scalingo deployment skill
- [x] UI best practices skill
- [x] Unsloth fine-tuning skill
- [ ] Testing patterns skill
- [ ] API design skill
- [ ] Database design skill
- [ ] Microservices patterns skill
- [ ] Security best practices skill
- [ ] Performance optimization skill

## 🔗 Related Projects

- [Hugging Face SmolAgents](https://github.com/huggingface/smolagents) - Minimalist AI agent framework
- [LangChain](https://github.com/langchain-ai/langchain) - Building applications with LLMs
- [genai-tk](https://github.com/tclatos/genai-tk) - YAML-driven GenAI toolkit
- [Model Context Protocol](https://modelcontextprotocol.io/) - Standard for connecting AI systems
- [Scalingo](https://scalingo.com/) - European Platform-as-a-Service
- [Unsloth](https://github.com/unslothai/unsloth) - Fast LLM fine-tuning
- [Adaption](https://docs.adaptionlabs.ai/) - Synthetic data augmentation platform

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
