import os
import os.path as osp
import pyximport

pyximport.install()
from config import cfg
from lang_dict.lang import LanguageIndex
from net.net import *
from utils.img_utils import *
from utils import edit_distance as ed
from tqdm import *
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

label_lang = LanguageIndex()
vocab_size = len(label_lang.word2idx)

BATCH_SIZE = 1
embedding_dim = cfg.EMBEDDING_DIM
units = cfg.UNITS

encoder = Encoder(units, BATCH_SIZE)
decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)

checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def evaluate(encoder, decoder, img_path, label_lang):
    img = process_img(img_path)

    enc_output, enc_hidden = encoder(np.expand_dims(img, axis=0))

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([label_lang.word2idx['<start>']] * BATCH_SIZE, 1)

    results = np.zeros((BATCH_SIZE, 25), np.int32)

    for t in range(1, 25):
        # passing enc_output to the decoder
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

        predicted_id = tf.argmax(predictions, axis=-1).numpy()

        results[:, t - 1] = predicted_id

        dec_input = tf.expand_dims(predicted_id, 1)

    pred = [process_result(result, label_lang) for result in results]

    return pred


root = "../mnt/ramdisk/max/90kDICT32px"


def create_dataset_from_file(root, file_path):
    with open(osp.join(root, file_path), "r") as f:
        readlines = f.readlines()

    img_paths = []
    for img_name in tqdm(readlines, desc="read dir:"):
        img_name = img_name.rstrip().strip()
        img_name = img_name.split(" ")[0]
        img_path = root + "/" + img_name
        # if osp.exists(img_path):
        img_paths.append(img_path)
    img_paths = img_paths[:1000]
    labels = [img_path.split("/")[-1].split("_")[-2] for img_path in tqdm(img_paths, desc="generator label:")]
    return img_paths, labels


img_paths, labels = create_dataset_from_file(root, "annotation_valid.txt")

tqdm_range = tqdm(range(len(labels)))
all_acc = 0
all_distance = 0
for i in tqdm_range:
    img_path = img_paths[i]
    label = labels[i]
    pred = evaluate(encoder=encoder, decoder=decoder, img_path=img_path, label_lang=label_lang)
    acc = compute_accuracy([label], pred)
    all_acc += acc
    distance = ed.calculate_edit_distance(label, pred[0])
    all_distance += distance
    tqdm_des = "evaluate: {:d}/{:d} acc = {:f} edit distance= {:f}".format(i + 1, len(labels), all_acc / (i + 1),
                                                                           all_distance / (i + 1))
    tqdm_range.set_description(tqdm_des)
