def bionify_text(text: str, ratio: float = 0.5) -> str:
    """
    Transforms the input text into a 'bionic reading' style by bolding the first half of each word.
    Args:
        text (str): The input text to be transformed.
    Returns:
        str: The transformed text with bionic reading style.
    """
    def bionic_word(word: str) -> str:
        if not word:
            return word
        bold_len = max(1, round(len(word) * ratio))
        return f"**{word[:bold_len]}**{word[bold_len:]}"

    words = text.split()
    bionic_words = [bionic_word(word) for word in words]
    return ' '.join(bionic_words)