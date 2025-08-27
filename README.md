# Research Summarizer (CLI)

A small command-line tool that:
- Takes a topic from you,
- Searches the web for relevant sources,
- Iteratively summarizes and refines the findings,
- Saves the final result to a Markdown file that includes a sources section.

It uses an OpenAI-compatible chat model endpoint (local or remote) and performs web search to gather context before summarizing.

## Quickstart

Prerequisites:
- Python 3.11+
- uv (package/venv manager)

Setup:
1) Create and activate a virtual environment
   - macOS/Linux:
     - uv venv
     - source .venv/bin/activate
   - Windows (PowerShell):
     - uv venv
     - .\.venv\Scripts\Activate.ps1

2) Install dependencies
   - uv pip install -r requirements.txt

3) Configure environment
   - Copy the example file and edit values as needed:
     - cp .env.example .env
   - At minimum, confirm these:
     - BASE_URL: OpenAI-compatible API base URL (e.g., a local server)
     - LLM_MODEL: Model name known to your API
     - MAX_WEB_RESEARCH_LOOPS: How many research iterations to perform
     - SEARCH_API: Web search backend (default: duckduckgo)
     - FETCH_FULL_PAGE: Whether to fetch full page content for sources (true/false)
     - STRIP_THINKING_TOKENS: Whether to remove <think>…</think> content if your model returns it (true/false)

## Running

You can run the tool in any of the following ways after activating the venv:

- Python module:
  - python -m src.summary_graph

- Direct script:
  - python src/summary_graph.py

- Using uv to run:
  - uv run python -m src.summary_graph

You’ll be prompted for a topic, for example:
- Give me a topic to research: modern data pipeline orchestration

The program will search the web, summarize and refine, then write the result to a Markdown file.

## Output

- A file named like:
  - summary_<your topic>_<unique id>.md
- The content includes:
  - A concise summary of the topic based on gathered sources
  - A sources section with links to references used

## Configuration Details

Set configuration with environment variables (via .env or shell):
- BASE_URL: OpenAI-compatible API (e.g., http://127.0.0.1:1234/v1)
- LLM_MODEL: Model name available at the BASE_URL endpoint
- LLM_PROVIDER: Provider label (defaults to openai-style compatibility)
- MAX_WEB_RESEARCH_LOOPS: Integer research loop count (e.g., 3)
- SEARCH_API: duckduckgo
- FETCH_FULL_PAGE: true/false to include full page content (can be slower/heavier)
- STRIP_THINKING_TOKENS: true/false to remove hidden reasoning tags from responses

Tips:
- If you’re using a local model server, ensure it’s running at BASE_URL and that LLM_MODEL matches an available model.
- If web pages are large or fetching times out, set FETCH_FULL_PAGE=false.

## Troubleshooting

- Connection errors to the model API:
  - Verify BASE_URL is correct and the server is running.
  - Confirm the model name in LLM_MODEL is available.

- Empty or sparse results:
  - Increase MAX_WEB_RESEARCH_LOOPS for deeper iterations.
  - Try a more specific topic or rerun to get fresh search results.

- Slow runs:
  - Disable FETCH_FULL_PAGE or reduce MAX_WEB_RESEARCH_LOOPS.

## License

This repository is provided as-is. See LICENSE if present.
