from datetime import datetime

# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")

query_writer_instructions ="""
Your task: craft a precise web search query tailored to the topic.

<CONTEXT>
Current date: {current_date}
Make the query time-aware and favor the most current, authoritative information available as of this date.
</CONTEXT>

<TOPIC>
{research_topic}
Focus the query on the core intent of this topic. Add clarifiers (key entities, use case, domain/industry, version/standard, geography, timeframe like “2024–{current_date}”, or site/filetype filters) to improve relevance and reduce ambiguity.
</TOPIC>

<FORMAT>
Return a JSON object with exactly these keys:
- "query": A concise, targeted search string optimized for high-signal results
- "rationale": Briefly explain how the query’s scope, qualifiers, and recency improve relevance
</FORMAT>

<EXAMPLE>
Example output:
{{
    "query": "machine learning transformer architecture explained",
    "rationale": "Understanding the fundamental structure of transformer models"
}}
</EXAMPLE>

Respond only with JSON:"""