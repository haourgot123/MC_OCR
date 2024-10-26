import torch
from PIL import Image, ImageDraw
import numpy as np
import cv2
import pandas as pd
from CRAFT import CRAFTModel, draw_boxes, draw_polygons, boxes_area, polygons_area


def rotation_image(boxes, img):
    # Compute angle for rotation
    boxes_ratio = []
    direct_roate = []
    total_width = []
    total_height = []
    for box in boxes:
        x1, y1, x2, y2 = box[0][0], box[0][1], box[1][0], box[1][1]
        total_width.append(x2 - x1)
        total_height.append(y2 - y1)
        box_ratio = abs(y1 - y2)  / abs(x1 - x2)
        direct_roate.append(y1 - y2)
        boxes_ratio.append(box_ratio)
    mean_ratio = np.mean(boxes_ratio)
    mean_direct_rotate = np.mean(direct_roate)
    angle = np.arctan(mean_ratio)
    if mean_direct_rotate >= 0:
        angle_degree = -np.degrees(angle)
    else:
        angle_degree  = np.degrees(angle)
    # Rotate image
    rotated_image = img.rotate(angle_degree, expand = True)
    if np.mean(total_width) < np.mean(total_height):
        rotated_image = img.rotate(90, expand = True)
    # rotated_image.save('abc.jpg')
    return rotated_image




class TextDetection():
    def __init__(self, device = 'cuda'):
        self.device = device
        self.model = CRAFTModel(self.device, use_refiner = True, fp16 = True)
    def predict_boxes(self, img):
        boxes = self.model.get_boxes(img)
        return boxes
    def draw_boxes(self, boxes, img):
        draw = ImageDraw.Draw(img)
        for box in boxes:
            x1, y1, x2, y2 = box[0][0], box[0][1],box[2][0], box[2][1]
            if x1 >= x2 or y1 >= y2:
                print(f'Exception: {x1, y1, x2, y2}')
                continue
            draw.rectangle((x1, y1, x2, y2), outline='red')
        img.save('rotated_image.jpg')

# Test

# if __name__ == '__main__':
#     img_path = 'test/mcocr_val_145114ixmyt.jpg'
#     img = Image.open(img_path)
#     text_det = TextDetection(device = 'cuda')
#     boxes = text_det.predict_boxes(img)
#     # Plot result
#     rotated_img = rotation_image(boxes, img)
#     boxes = text_det.predict_boxes(rotated_img)
#     text_det.draw_boxes(boxes, rotated_img)
   

