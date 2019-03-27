import cv2
import numpy as np
from typing import List


def process_img(img_path):
    imread = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    imread = resize_image(imread, 100, 32)
    imread = np.expand_dims(imread, axis=-1)
    imread = np.array(imread, np.float32)
    return imread

def resize_image(image, out_width, out_height):
    """
        Resize an image to the "good" input size
    """
    im_arr = image
    h, w = np.shape(im_arr)[:2]
    ratio = out_height / h

    im_arr_resized = cv2.resize(im_arr, (int(w * ratio), out_height))
    re_h, re_w = np.shape(im_arr_resized)[:2]

    if re_w >= out_width:
        final_arr = cv2.resize(im_arr, (out_width, out_height))
    else:
        final_arr = np.ones((out_height, out_width), dtype=np.uint8) * 255
        final_arr[:, 0:np.shape(im_arr_resized)[1]] = im_arr_resized
    return final_arr


def preprocess_label(label):
    label = label.rstrip().strip()
    w = '<start> '
    for i in label:
        w += i + ' '
    w += ' <end>'
    return w


def process_result(result, label_lang):
    result_label = ""
    for i in result:
        if label_lang.idx2word[i] != '<end>':
            result_label += label_lang.idx2word[i]
        else:
            return result_label
    return result_label


def compute_accuracy(ground_truth: List[str], predictions: List[str]) -> np.float32:
    accuracy = []
    for index, label in enumerate(ground_truth):
        prediction = predictions[index]
        total_count = len(label)
        correct_count = 0
        try:
            for i, tmp in enumerate(label):
                if tmp == prediction[i]:
                    correct_count += 1
        except IndexError:
            continue
        finally:
            try:
                accuracy.append(correct_count / total_count)
            except ZeroDivisionError:
                if len(prediction) == 0:
                    accuracy.append(1)
                else:
                    accuracy.append(0)

    accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
    return accuracy
