# Imports
import numpy as np
from PIL import Image,ImageDraw,ImageFont

# Write the required arguments

# The function should plot the predicted boxes on the images and save them.
# Tip: keep the dimensions of the output image less than 800 to avoid RAM crashes.
# from PIL.ImageDraw import ImageDraw
# from PIL.ImageFont import ImageFont


def plot_boxes(image, pred_boxes, pred_classes, pred_scores, output_path):  # image is in 3*W*H
    indices = []
    len_list = len(pred_scores)
    if len_list > 5:
        pred_scores_copy = pred_scores.copy()
        pred_scores_copy.sort(reverse=True)
        for i in range(5):
            indices.append(pred_scores.index(pred_scores_copy[i]))

    else:
        for i in range(len_list):
            indices.append(i)

    for i in indices:
        (y1, x1), (y2, x2) = pred_boxes[i]
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        y1 = min(y1, image.shape[2] - 1)
        y2 = min(y2, image.shape[2] - 1)
        x1 = min(x1, image.shape[1] - 1)
        x2 = min(x2, image.shape[1] - 1)
        img = image.copy()
        for x in (x1, x2):
            for y in range(y1, y2):
                img[0][x][y] = 1
                img[1][x][y] = 0
                img[2][x][y] = 0
        for y in (y1, y2):
            for x in range(x1, x2):
                img[0][x][y] = 1
                img[1][x][y] = 0
                img[2][x][y] = 0

    img = img.transpose((1, 2, 0))  # converting to W*H*3
    imgPIL = Image.fromarray((img * 255).astype(np.uint8))
    imgDraw = ImageDraw.Draw(imgPIL)
    font = ImageFont.load_default()
    for i in indices:
        (y1, x1), (y2, x2) = pred_boxes[i]
        imgDraw.text((y1, x2), pred_classes[i], font=font, fill=(255, 0, 0))
    imgPIL.save(output_path)
    return img
