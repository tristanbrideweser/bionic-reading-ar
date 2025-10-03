from src.ocr.ocr_engine import load_image, preprocess_image, extract_text
from src.utils.utils import read_file, ocr_accuracy, save_two_column_md as save_to_md, display_md_file as display
from src.bionic.bionic import bionify_text
import cv2

def main():
    image = load_image("src/tests/hunger_games_ocr.jpeg")

    pre = preprocess_image(image)
    
    extracted_text = extract_text(pre)

    extracted_text_lines = extracted_text.splitlines()

    bionic_text = "\n".join([bionify_text(line) for line in extracted_text_lines])
    
    real_text = read_file("src/tests/hunger_games_ocr.txt")

    accuracy = round(ocr_accuracy(extracted_text, real_text),3) * 100

    file_path = "src/tests/ocr_results_files/ocr_results.md"

    save_to_md(file_path, 
               real_text, 
               bionic_text, 
               accuracy)
    
    display(file_path)
    
if __name__ == "__main__":
    main()