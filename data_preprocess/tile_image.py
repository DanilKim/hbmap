from glob import glob
import os
import argparse
import numpy as np
from PIL import Image


def tile_image(p_img, folder, size: int) -> list:
    w = h = size
    im = np.array(Image.open(p_img))
    # https://stackoverflow.com/a/47581978/4521646
    tiles = [im[i:(i + h), j:(j + w), ...] for i in range(0, im.shape[0], h) for j in range(0, im.shape[1], w)]
    idxs = [(i, (i + h), j, (j + w)) for i in range(0, im.shape[0], h) for j in range(0, im.shape[1], w)]
    name, _ = os.path.splitext(os.path.basename(p_img))
    files = []
    for k, tile in enumerate(tiles):
        if tile.shape[:2] != (h, w):
            tile_ = tile
            tile = np.zeros_like(tiles[0])
            tile[:tile_.shape[0], :tile_.shape[1], ...] = tile_
        p_img = os.path.join(folder, f"{name}_{k:03}.png")
        Image.fromarray(tile).save(p_img)
        files.append(p_img)
    return files, idxs


def get_args():
    parser = argparse.ArgumentParser(description="Process images to tile")
    parser.add_argument('--type', required=True, choices=["images", "masks"])
    parser.add_argument('--size', required=True, type=int, choices=[256, 512, 1024])
    return parser.parse_args()


if __name__=="__main__":
    args = get_args()

    image_dir = f"/data/train_{args.type}"
    tile_dir = f"/data/train_{args.type}_{args.size}tile"
    format = "tiff" if args.type=="images" else "png"

    if not os.path.isdir(tile_dir):
        os.makedirs(tile_dir)
    for index, filename in enumerate(glob(os.path.join(image_dir, '*'))):
        files, idxs = tile_image(filename, tile_dir, args.size)
        if (index + 1) % 10 == 0:
            print(index + 1)
