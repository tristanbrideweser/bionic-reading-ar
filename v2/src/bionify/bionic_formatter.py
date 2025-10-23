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

"""
bionic_formatter.py
-------------------
Implements the transformation of plain text into "bionic reading" format,
where key portions of each word are bolded for faster reading.
Includes caching for performance.
"""

import re
from typing import Dict, Tuple


class BionicFormatter:
    """Handles bionic text formatting with caching."""
    
    def __init__(self, emphasis_ratio: float = 0.4):
        """
        Initialize the formatter.
        
        Args:
            emphasis_ratio: Portion of each word to emphasize (0-1).
        """
        self.emphasis_ratio = emphasis_ratio
        self._cache: Dict[str, str] = {}
    
    def format_text(self, text: str) -> str:
        """
        Convert text to bionic format with HTML-style bold tags.
        Uses cache for performance.
        
        Args:
            text: Plain text to format
            
        Returns:
            Text with <b></b> tags around emphasized portions
        """
        # Check cache first
        if text in self._cache:
            return self._cache[text]
        
        # Split into words while preserving punctuation
        words = re.findall(r'\b\w+\b|\W+', text)
        formatted_words = []
        
        for word in words:
            if re.match(r'\w+', word):  # It's a word
                formatted_words.append(self._bionify_word(word))
            else:  # It's whitespace or punctuation
                formatted_words.append(word)
        
        result = ''.join(formatted_words)
        
        # Cache the result
        self._cache[text] = result
        return result
    
    def _bionify_word(self, word: str) -> str:
        """
        Apply emphasis to the first part of a word.
        
        Args:
            word: Single word to format
            
        Returns:
            Word with <b></b> tags around first portion
        """
        if len(word) <= 1:
            return f"<b>{word}</b>"
        
        # Calculate emphasis length (minimum 1, maximum word length)
        emphasis_len = max(1, int(len(word) * self.emphasis_ratio))
        
        # Split word into emphasized and normal parts
        emphasized = word[:emphasis_len]
        normal = word[emphasis_len:]
        
        return f"<b>{emphasized}</b>{normal}"
    
    def format_for_overlay(self, formatted_text: str) -> Tuple[str, str]:
        """
        Split formatted text into bold and normal portions for OpenCV rendering.
        
        Args:
            formatted_text: Text with <b></b> tags
            
        Returns:
            Tuple of (bold_parts, normal_parts) as strings
        """
        # Extract bold text
        bold_pattern = r'<b>(.*?)</b>'
        bold_parts = ''.join(re.findall(bold_pattern, formatted_text))
        
        # Extract normal text
        normal_parts = re.sub(bold_pattern, '', formatted_text)
        normal_parts = re.sub(r'<.*?>', '', normal_parts)
        
        return bold_parts, normal_parts
    
    def clear_cache(self):
        """Clear the formatting cache."""
        self._cache.clear()
    
    def get_cache_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


def format_bionic_text(text: str, emphasis_ratio: float = 0.4) -> str:
    """
    Quick utility function for one-off formatting.
    For repeated use, create a BionicFormatter instance to use caching.
    
    Args:
        text: The input plain text
        emphasis_ratio: Portion of each word to emphasize (0-1)
        
    Returns:
        Formatted string with <b></b> tags
    """
    formatter = BionicFormatter(emphasis_ratio)
    return formatter.format_text(text)


if __name__ == "__main__":
    # Test the formatter
    formatter = BionicFormatter()
    
    samples = [
        "Hello world",
        "Computer vision makes machines see text",
        "The quick brown fox jumps over the lazy dog"
    ]
    
    for sample in samples:
        formatted = formatter.format_text(sample)
        print(f"Original: {sample}")
        print(f"Bionic:   {formatted}")
        print()
    
    print(f"Cache size: {formatter.get_cache_size()}")