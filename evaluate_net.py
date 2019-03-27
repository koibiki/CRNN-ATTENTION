import os

from config import cfg
from lang_dict.lang import LanguageIndex
from net.net import *
from utils.img_utils import *

label_lang = LanguageIndex()
vocab_size = len(label_lang.word2idx)

BATCH_SIZE = 1
embedding_dim = cfg.EMBEDDING_DIM
units = cfg.UNITS

encoder = Encoder(units, BATCH_SIZE)
decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)

global_step = tf.train.get_or_create_global_step()

start_learning_rate = cfg.LEARNING_RATE
learning_rate = tf.Variable(start_learning_rate, dtype=tf.float32)

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

    preds = [process_result(result, label_lang) for result in results]

    print("real :" + preds[0])


img_path = "./sample/1_bridleway_9530.jpg"

evaluate(encoder=encoder, decoder=decoder, img_path=img_path, label_lang=label_lang)
