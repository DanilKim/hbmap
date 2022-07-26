import argparse
import random
import glob
import argparse


def split_train_val(val_percent):
    filenames = []
    for filename in glob.glob('/data/train_images/*.tiff'):
        filenames.append(filename.split('/')[-1].split('.')[0])

    # split
    random.shuffle(filenames)
    n_val = int(len(filenames) * val_percent)
    train_list = filenames[n_val:]
    val_list = filenames[:n_val]
    
    # write train file
    with open('/data/train.txt', 'w') as f:
        for filename in train_list:
            f.write(filename+'\n')

    # write val file
    with open('/data/val.txt', 'w') as f:
        for filename in val_list:
            f.write(filename+'\n')
    return
    


def get_args():
    parser = argparse.ArgumentParser(description="Process images to tile")
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--val_percent', type=float, default=0.1)
    return parser.parse_args()


if __name__=="__main__":
    args = get_args()
    random.seed(args.random_seed)

    split_train_val(val_percent = args.val_percent)