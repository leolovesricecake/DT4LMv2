"""
Word Swap by swapping synonyms in WordNet
------------------------------------------------
"""

import nltk
from nltk.corpus import wordnet

import textattack

from .word_swap import WordSwap


class WordSwapWordNet(WordSwap):
    """Transforms an input by replacing its words with synonyms provided by
    WordNet.

    >>> from textattack.transformations import WordSwapWordNet
    >>> from textattack.augmentation import Augmenter

    >>> transformation = WordSwapWordNet()
    >>> augmenter = Augmenter(transformation=transformation)
    >>> s = 'I am fabulous.'
    >>> augmenter.augment(s)
    """

    def __init__(self, language="eng"):
        # Experiment startup must never perform an implicit network request.
        # WordNet is prepared explicitly so offline runs fail immediately with
        # an actionable command instead of waiting in NLTK's downloader.
        try:
            wordnet.ensure_loaded()
        except LookupError as exc:
            raise LookupError(
                "WordSwapWordNet requires the local NLTK WordNet corpus. "
                "Prepare it before the experiment with: "
                "python -m nltk.downloader wordnet"
            ) from exc
        if language != "eng":
            try:
                nltk.data.find("corpora/omw-1.4")
            except LookupError as exc:
                raise LookupError(
                    "Non-English WordSwapWordNet requires the local omw-1.4 "
                    "corpus. Prepare it with: "
                    "python -m nltk.downloader omw-1.4"
                ) from exc
        if language not in wordnet.langs():
            raise ValueError(f"Language {language} not one of {wordnet.langs()}")
        self.language = language

    def _get_replacement_words(self, word, random=False):
        """Returns a list containing all possible words with 1 character
        replaced by a homoglyph."""
        synonyms = set()
        for syn in wordnet.synsets(word, lang=self.language):
            for syn_word in syn.lemma_names(lang=self.language):
                if (
                    (syn_word != word)
                    and ("_" not in syn_word)
                    and (textattack.shared.utils.is_one_word(syn_word))
                ):
                    # WordNet can suggest phrases that are joined by '_' but we ignore phrases.
                    synonyms.add(syn_word)
        return list(synonyms)
