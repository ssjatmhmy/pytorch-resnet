'''Create a val dataset that can be read by torchvision.datasets.ImageFolder
   folder structure:
   ./ILSVRC2012/
     -- synset_words.txt
     -- val.txt
     -- train/
     -- val/
'''

import os
from shutil import copyfile

def mkdir_p(path):
    """make dir if not exist"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


datadir = 'val_torch'
mkdir_p(datadir)

idx2label = dict()

with open('synset_words.txt', 'r') as f:
    for idx, line in enumerate(f.readlines()):
        label = line.strip().split()[0]
        idx2label[idx] = label
        path = os.path.join(datadir, label)
        mkdir_p(path)

with open('val.txt', 'r') as f:
    for line in f.readlines():
        filename, idx = line.strip().split()
        idx = int(idx)
        label = idx2label[idx]
        src_path = os.path.join('val', filename)
        dst_path = os.path.join(datadir, label, filename)
        copyfile(src_path, dst_path)
