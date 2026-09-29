import re
import json
import numpy as np
from logger import log_msg
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from constants import (
    EMBEDDING_MODEL_NAME,
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    MAX_RESULT_DEFAULT,
    KNOWLEDGE_FOLDER_NAME,
    Tier,
)

# Define the schemas of chunk and search result.
@dataclass
class Chunk:
    text: str
    label: str
    source_file: str

@dataclass
class SearchResult:
    tier: Tier
    matches: List[Tuple[Chunk, float]]

def load_json_chunks(path: Path) -> List[Chunk]:
    """
    Utility function to read and parse a JSON file into a chunks list.
    Loads a JSON array of objects from a file, into chunks.
    """
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"WARN: skipping {path.name}, could not load as JSON due to error: {error}")
        return []
    
    # Expects the JSON to be an array of objects.
    if not isinstance(records, list):
        print(f"WARN: skipping {path.name}, as the program expects an array of objects as the JSON.")
        return []

    chunks = []
    
    for record in records:
        if not isinstance(record, dict) or not record: # Expects each record item to be an object, representing one product
            continue
        label = str(record["name"]) # Expects each object in the JSON contains a "name" property.
        lines = [f"{key}: {value}" for key, value in record.items()]
        chunks.append(Chunk(text="\n".join(lines), label=label, source_file=path.name))

    log_msg(f"load_json_chunks() generated {len(chunks)} JSON chunks in total.", True)
    return chunks

def load_txt_chunks(path: Path) -> List[Chunk]:
    """
    Utility function to read and parse a text file into a list of chunks.
    Splits on blank lines (paragraph breaks), one chunk per paragraph.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as error:
        print(f"WARN: skipping {path.name}, could not read file due to error: {error}")
        return []

    paragraphs = re.split(r"\n\s*\n", content.strip()) # Matches at least one blank line, between each paragraph
    chunks = []
    
    for paragraph_index, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        label = f"{path.stem} #{paragraph_index + 1}"
        chunks.append(Chunk(text=paragraph, label=label, source_file=path.name))

    log_msg(f"load_txt_chunks() generated {len(chunks)} TXT chunks in total.", True)
    return chunks


def load_knowledge_base() -> List[Chunk]:
    """
    Function merger to call the "load_json_chunks" & "load_txt_chunks" functions, to produce a total list of chunks.
    """
    knowledge_folder = Path(KNOWLEDGE_FOLDER_NAME)
    if not knowledge_folder.is_dir():
        raise FileNotFoundError(f"Knowledge folder not found: {knowledge_folder}")

    chunks: List[Chunk] = []
    
    for path in sorted(knowledge_folder.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".json":
            chunks.extend(load_json_chunks(path))
        elif path.suffix == ".txt":
            chunks.extend(load_txt_chunks(path))
        else:
            print(f"WARN: skipping {path.name}. Only TXT/JSON are currently supported!")

    if not chunks:
        raise ValueError(f"No usable knowledge chunks found in {knowledge_folder}")

    log_msg(f"load_knowledge_base() produced {len(chunks)} chunks in total.", True)
    return chunks


class KnowledgeIndex:
    """
    In-Memory embedding index for the knowledge base contents.
    The "build" function MUST be called to construct the embeddings, before doing any searches.
    The "search" function CANNOT be called if the "build" function is never called before.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self._embedding_model = SentenceTransformer(model_name)
        self._chunks: List[Chunk] = []
        self._embeddings: Optional[np.ndarray] = None

    def build(self) -> None:
        self._chunks = load_knowledge_base()
        chunk_texts = [chunk.text for chunk in self._chunks]
        self._embeddings = self._embedding_model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=False)
        log_msg("KnowledgeIndex instance build() process finishes.", True)

    def search(self, query: str, max_results = MAX_RESULT_DEFAULT) -> SearchResult:
        if self._embeddings is None:
            raise RuntimeError("Error: The knowledge embeddings has not been built yet!")
        if not query.strip():
            return SearchResult(tier="no_match", matches=[])

        query_embedding = self._embedding_model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]

        scores = self._embeddings @ query_embedding # A list of score numbers, per chunk
        # The sorted chunk indices, based on matching score from high to low
        ranked_indices = np.argsort(scores)[::-1]
        # Create a list of (chunk, score) tuple pairs, best match first, for every chunk.
        ranked = [(self._chunks[index], float(scores[index])) for index in ranked_indices]

        # Based on the high/low confidence parameter, return the first "max_results" of matches
        # 1. Good matches
        confident_matches = [(chunk, score) for chunk, score in ranked if score >= HIGH_CONFIDENCE_THRESHOLD][:max_results]
        if confident_matches:
            log_msg(f"search() finishes with CONFIDENT matches. {[(chunk.label, score) for chunk, score in confident_matches]}", False)
            return SearchResult(tier="confident", matches=confident_matches)

        # 2. Mediocre matches
        ambiguous_matches = [(chunk, score) for chunk, score in ranked if score >= LOW_CONFIDENCE_THRESHOLD][:max_results]
        if ambiguous_matches:
            log_msg(f"search() finishes with AMBIGUOUS matches. {[(chunk.label, score) for chunk, score in ambiguous_matches]}", False)
            return SearchResult(tier="ambiguous", matches=ambiguous_matches)

        # 3. Bad matches returns nothing
        log_msg(f"search() finishes with NO match.", False)
        return SearchResult(tier="no_match", matches=[])
