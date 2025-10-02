from src.ocr.ocr_processor import load_image, preprocess_image, extract_text
import cv2

def main():
    image = load_image("src/tests/hunger_games_ocr.jpeg")
    pre = preprocess_image(image)
    #cv2.imshow("preprocess", pre)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    text = extract_text(pre)
    print(text)

if __name__ == "__main__":
    main()