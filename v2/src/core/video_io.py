from PIL import Image, ImageDraw, ImageFont
import numpy as np


def overal_bionic_text(frame, box, styled_words, bold_font, regular_font, color=(0,0,0)):
    """
    Draws the bionic reading text on the frame inside the given box.

    Args:
        frame: OpenCV BGR numpy array
        box: 4-point bounding box around text (list of [x, y])
        styled_words: List of (bold_part, rest_part) tuples
        bold_font: PIL ImageFont instance (bold)
        regular_font: PIL ImageFont instance (regular)
        color: Text color in RGB
    """
    # Convert frame to PIL Image for text drawing
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # For simplicity, calculate start x,y from box (e.g. top-left)
    x, y = box[0][0], box[0][1]

    # Draw each word, bold part then rest part
    for bold, rest in styled_words:
        draw.text((x, y), bold, font=bold_font, fill=color)
        x += draw.textlength(bold, font=bold_font)
        draw.text((x, y), rest + " ", font=regular_font, fill=color)
        x += draw.textlength(rest + " ", font=regular_font)

    # Convert back to OpenCV BGR format
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)