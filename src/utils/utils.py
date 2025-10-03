from difflib import SequenceMatcher
import subprocess
from pathlib import Path
import cv2

def read_file(filename: str) -> str:
    """
    Reads the content of a file and returns it as a string.
    Args:
        filename (str): Path to the file.
    Returns:
        str: Contents of the file.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return f"Error: File '{filename}' not found."
    except Exception as e:
        return f"Error: {e}"
    

def ocr_accuracy(extracted_text: str, real_text: str) -> float:
    """
    Compares the extracted text with the real text and returns an accuracy score.
    Args:
        extracted_text (str): Text extracted by OCR.
        real_text (str): Ground truth text.
    Returns:
        float: Similarity ratio between 0 and 1.
    """
    return SequenceMatcher(None, extracted_text.split(), real_text.split()).ratio()


def save_two_column_md(filename: str, original_text: str, bionic_text: str, accuracy: float) -> None:
    """
    Save original and bionic text side-by-side in a markdown file using an HTML table.
    """
    # Split texts into lines to preserve paragraphs
    original_lines = original_text.splitlines()
    bionic_lines = bionic_text.splitlines()

    # Ensure both lists have the same length
    max_len = max(len(original_lines), len(bionic_lines))
    original_lines += [""] * (max_len - len(original_lines))
    bionic_lines += [""] * (max_len - len(bionic_lines))

    # Build HTML table
    table_rows = ""
    for orig, bio in zip(original_lines, bionic_lines):
        table_rows += f"<tr><td>{orig}</td><td>{bio}</td></tr>\n"

    html_table = f"""
# OCR Results
Accuracy: {accuracy}%
<table>
<tr>
<th>Original Text</th>
<th>Bionic Reading</th>
</tr>
{table_rows}
</table>
"""

    # Save to file
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_table)
        print(f"Saved two-column Markdown to {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")


def display_md_file(filename: str) -> None:
    """
    Previews the given Markdown file using Quarto.
    Args:
        filename (str): Path to the markdown file.
    """
    path = Path(filename)
    if path.exists():
        try:
            # Run Quarto preview command
            subprocess.run([
                "quarto", "preview", str(path),
                "--no-browser", "--no-watch-inputs"
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Quarto preview: {e}")
    else:
        print(f"File {filename} does not exist.")


def show_image(window_name: str, image) -> None:
    """
    Displays an image in a window.
    Args:
        window_name (str): Name of the window.
        image: Image to be displayed (numpy array).
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
