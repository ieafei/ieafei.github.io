# ieafei's Learning Hub

> 🌐 **Live Site**: [https://ieafei.github.io](https://ieafei.github.io)

A personal learning website to organize and share study notes on AI/ML infrastructure, frameworks, and computer graphics.

## 📁 Directory Structure

```
ieafei.github.io/
├── index.html              # Homepage (module navigation)
├── style.css               # Global styles
├── _config.yml             # GitHub Pages config
├── README.md
└── notes/                  # All study notes
    ├── vllm/               # ⚡ vLLM related notes
    │   └── vllm_generate_flow.html
    ├── langchain/          # 🦜 LangChain related notes
    └── dgp/                # 🔺 Digital Geometry Processing notes
```

## 📚 Learning Modules

| Module | Topic | Status |
|--------|-------|--------|
| ⚡ **vLLM** | LLM inference engine internals — architecture, scheduling, PagedAttention, speculative decoding | Active |
| 🦜 **LangChain** | LLM application framework — chains, agents, RAG, memory, tool integration | Coming Soon |
| 🔺 **DGP** | Digital Geometry Processing — mesh processing, surface parameterization, discrete differential geometry | Coming Soon |

## 🚀 Local Development

```bash
# Serve locally with Python
cd ieafei.github.io
python3 -m http.server 8080

# Then open http://localhost:8080
```

## 🛠️ Tech Stack

- Pure static HTML + CSS (no build tools required)
- Deployed via GitHub Pages
- Dark theme with responsive design

## 📝 How to Add Notes

1. Create an HTML file in the appropriate `notes/{category}/` directory
2. Add a link entry in `index.html` under the corresponding module section
3. Use `../../index.html` for breadcrumb/back links in note pages

## 📎 Related Repositories

- [vllm-study](https://github.com/ieafei/vllm-study) — Detailed vLLM source code study notes (separate site)
