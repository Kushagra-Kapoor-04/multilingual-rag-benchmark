"""
embeddings/embedder_factory.py

Central place to select the correct embedding model based on language
or config, so the rest of the pipeline never hardcodes a specific
embedding model — it just asks the factory for "the embedder for hi"
or "the embedder for en".
"""

from embeddings.base_embedder import BaseEmbedder

# Cache loaded embedders so we don't reload the same model multiple times
# in one process (model loading is the slow part).
_embedder_cache = {}


def get_embedder(language: str) -> BaseEmbedder:
    """
    Return the appropriate embedder instance for a given language code.

    Args:
        language: "en" for English, "hi" for Hindi.

    Returns:
        An initialized embedder instance implementing BaseEmbedder.

    Raises:
        ValueError: if the language is not yet supported.
    """
    language = language.lower().strip()

    if language in _embedder_cache:
        return _embedder_cache[language]

    if language == "en":
        from embeddings.minilm_embedder import MiniLMEmbedder
        embedder = MiniLMEmbedder()
    elif language == "hi":
        from embeddings.muril_embedder import MurilEmbedder
        embedder = MurilEmbedder()
    else:
        raise ValueError(
            f"No embedder configured for language '{language}'. "
            f"Supported: 'en', 'hi'. (Code-mixed support planned for Major Project.)"
        )

    _embedder_cache[language] = embedder
    return embedder


if __name__ == "__main__":
    # Manual sanity check — run: python embeddings/embedder_factory.py
    en_embedder = get_embedder("en")
    hi_embedder = get_embedder("hi")

    print(f"English embedder: {en_embedder.model_name} (dim={en_embedder.embedding_dim})")
    print(f"Hindi embedder: {hi_embedder.model_name} (dim={hi_embedder.embedding_dim})")

    # Confirm caching works — should return the same instance, not reload
    en_embedder_again = get_embedder("en")
    assert en_embedder is en_embedder_again, "Caching failed — model was reloaded!"
    print("Caching check passed.")