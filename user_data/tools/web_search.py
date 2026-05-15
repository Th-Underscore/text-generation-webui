from modules.web_search import perform_web_search

tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web. Returns a list of results with title, URL, and snippet (short text excerpt). The snippet often answers the query directly. Use fetch_webpage on a URL if you need the full page.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
            },
            "required": ["query"]
        }
    }
}


def execute(arguments):
    query = arguments.get("query", "")
    results = perform_web_search(query, num_pages=None, fetch_content=False)
    output = [{"title": r["title"], "url": r["url"], "snippet": r["snippet"]} for r in results]
    return output if output else [{"error": "No results found."}]
