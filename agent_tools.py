from pathlib import Path
from smolagents import tool
from knowledge_index import KnowledgeIndex
from constants import OUTPUT_FOLDER_NAME

_knowledge_index = KnowledgeIndex()

def initialize() -> None:
    _knowledge_index.build()

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the private knowledge base for information relevant to the user's
    question, and report back how confident the match is.

    IMPORTANT: pass the user's question exactly as they asked it. Do not
    rephrase, reword, shorten, or guess at what they "really" meant. Always
    search with their literal wording.

    This tool decides the confidence tier for you, do not second-guess it.
    Depending on the tier reported, respond to the user as follows:

    - "confident": The content below may include more than one matching
      entry. Answer using ONLY this content. If multiple entries are
      shown, use whichever are relevant to the question and ignore the
      rest. You may cite source labels. Do not add information from
      outside this content.
    - "ambiguous": None of the content below scored as a strong match. Still
      answer using ONLY this content, doing your best, but you MUST
      clearly flag to the user that this answer is uncertain and may be
      inaccurate, since no strong match was found. Do not present it as a
      solid, confident fact.
    - "no_match": Say plainly that the knowledge base has no relevant
      information for this question. Do not guess or use outside knowledge.

    Args:
        query: The user's question as a string, passed exactly as written.

    Returns:
        A block of text string reporting the confidence tier, plus either the
        matching content or nothing, depending on the tier.
    """
    result = _knowledge_index.search(query)

    if result.tier == "no_match":
        return "CONFIDENCE: no_match\nNo relevant information was found in the knowledge base."

    matches_text = "\n\n".join(
        f"SOURCE: {chunk.label} ({chunk.source_file})\nCONTENT:\n{chunk.text}"
        for chunk, _ in result.matches
    )

    if result.tier == "ambiguous":
        return f"CONFIDENCE: ambiguous\n{matches_text}"

    return f"CONFIDENCE: confident\n{matches_text}"


@tool
def save_artifact(filename: str, content: str) -> str:
    """
    Save generated text (e.g. a summary the user asked for) to a file in the
    output folder, for the user to consume later.

    Args:
        filename: Desired file name only, e.g. "summary.md", "result.txt", or etc. 
            Any folder path included will be ignored. Files are always saved directly
            in the output folder.
        content: The full text content to write into that file.

    Returns:
        A confirmation message with the saved file's path, or an error
        message if the file could not be saved.
    """
    # Use .name to get only the LAST piece of the file name, stripping away any other path-like strings.
    # This is to avoid any potential path traversal security risk, making the agent ONLY able to write in "output" folder.
    safe_name = Path(filename).name
    if not safe_name:
        return "Could not save: no valid filename was given."

    output_path = Path(OUTPUT_FOLDER_NAME).joinpath(safe_name)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
    except OSError as error:
        return f"Could not save {safe_name} due to: {error}"

    return f"Artifact is saved to {output_path}"
