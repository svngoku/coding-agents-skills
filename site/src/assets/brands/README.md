# Library Marks

These marks identify the libraries and platforms covered by the gallery. They
are bundled locally, not fetched from third-party logo services at runtime.
Sources checked on 2026-10-25.

| File / skill | Mark | Official asset source | Provenance |
| --- | --- | --- | --- |
| `adaption-ai.png` | Adaption | [Site favicon](https://adaptionlabs.ai/favicon.ico) | [Adaption](https://adaptionlabs.ai/), [documentation](https://docs.adaptionlabs.ai/) |
| `typesafe-ai.svg` | TypeSafe | [Light-background integration icon](https://raw.githubusercontent.com/typesafe-ai/n8n-nodes-typesafe-ai/main/nodes/TypeSafeAi/typeSafeAi.svg) | [Official TypeSafe integration](https://github.com/typesafe-ai/n8n-nodes-typesafe-ai/blob/main/nodes/TypeSafeAi/TypeSafeAi.node.ts) |
| `langchain.png` | LangChain | [Official icon](https://raw.githubusercontent.com/langchain-ai/docs/main/src/images/brand/langchain-icon.png) | [Brand guidelines](https://www.langchain.com/brand-assets) |
| `smolagents.svg` | Hugging Face | [Official color mark](https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg) | [Hugging Face brand assets](https://huggingface.co/brand) |
| `scalingo.png` | Scalingo | [Official 96px favicon](https://doc.scalingo.com/favicon/favicon-96x96.png) | [Official favicon declarations](https://github.com/Scalingo/documentation/blob/master/src/_includes/organisms/favicons.html) |
| `unsloth-hf-jobs.png` | Unsloth | [Official Studio favicon](https://raw.githubusercontent.com/unslothai/unsloth/main/studio/frontend/public/favicon.png) | [Unsloth repository](https://github.com/unslothai/unsloth) |
| `prime-rl.png` | Prime Intellect | [Official site icon](https://www.primeintellect.ai/icons/logo-icon.png) | [Prime Intellect](https://www.primeintellect.ai/), [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) |

TypeSafe's mark identifies the provider, not a separately verified Jev-specific
logo. SmolAgents uses the Hugging Face owner mark; prime-rl uses Prime Intellect's
mark. No distinct genai-tk mark was established, so it keeps its topic icon.

Original colors, proportions, and supplied tile shapes are preserved. Adaption's
32px favicon was converted from ICO to PNG. Prime Intellect's 800px PNG was
downsampled to 96px for this small UI. Neither was redrawn. Other images are
unchanged. Light backing tiles keep the dark marks legible in either gallery
theme; Prime Intellect's white mark uses a black backing tile. Scalingo's raster
asset avoids coupling its SVG's system-theme media query to the gallery's manual
theme control.

Marks and trademarks remain the property of their respective owners. They are
used for identification, without implying affiliation or endorsement. The
repository's MIT license does not grant trademark rights or relicense these
third-party brand assets. Follow the owners' applicable brand guidelines; no
blanket reuse permission is asserted here.

## Adding or Updating a Mark

Use an authentic first-party mark, record its source here, and name it after the
skill directory with an `.svg` or `.png` extension. `SkillIcon` discovers those
files at build time. Check both themes and 24px rendering, and inspect SVGs for
scripts, embedded documents, or external resources before bundling. Do not invent
logos for unbranded topics. Missing or failed images retain the category icon.
