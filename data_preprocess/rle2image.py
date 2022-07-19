import os
import pandas as pd
from PIL import Image
from os import path
import sys

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from utils.rle import rle_decode


if __name__=="__main__":
    save_dir = 'data/train_labels'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    df = pd.read_csv('data/train.csv')
    for index, row in df.iterrows():
        mask = rle_decode(row['rle'], [row['img_height'], row['img_height']])
        mask = Image.fromarray(mask)
        mask.convert('L').save(os.path.join(save_dir, f"{row['id']}.png"))
        if (index+1) % 10 == 0:
            print(index+1)
