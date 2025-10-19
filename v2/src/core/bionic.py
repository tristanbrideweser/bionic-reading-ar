# bionic.py

import re


def bionify_text(text: str, ratio: float = 0.5) -> list[tuple[str, str]]:
    """
    Splits each word into a (bold_part, rest_part) tuple based on ratio.
    For rendering with actual bold and regular fonts (not Markdown).
    
    Args:
        text (str): The input text.
        ratio (float): Proportion of the word to bold.
    
    Returns:
        List[Tuple[str, str]]: A list of tuples representing each word split.
    """
    def bionic_word(word: str) -> tuple[str, str]:
        match = re.match(r"(\w+)(\W*)", word)
        if not match:
            return word, ''
        core, punct = match.groups()
        bold_len = max(1, round(len(core) * ratio))
        return core[:bold_len], core[bold_len:] + punct

    
    words = text.split()
    return [bionic_word(word) for word in words]
