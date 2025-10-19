# ! pip install paddleocr
# ! pip install paddlepaddle

from paddleocr import PaddleOCR

model = PaddleOCR(use_angle_cls=True, lang='en')

def detect_text(frame):

    rgb_frame = frame[:, :, ::-1]

    results = model.ocr(rgb_frame, cls=True)

    boxes_texts = []

    for line in results[0]:
        box = line[0]
        text = line[1][0]
        confidence = line[1][1]
        boxes_texts.append((box, text))
    return boxes_texts