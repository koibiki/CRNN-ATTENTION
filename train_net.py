import os
import os.path as osp
import time

from tqdm import *

from config import cfg
from net.net import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def max_length(tensor):
    return max(len(t) for t in tensor)


class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = cfg.CHAR_VECTOR

        self.create_index()

    def create_index(self):
        self.word2idx['<pad>'] = 0
        self.word2idx['<start>'] = 1
        self.word2idx['<end>'] = 2
        self.word2idx[''] = 3
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 4

        for word, index in self.word2idx.items():
            self.idx2word[index] = word


# root = "../mnt/ramdisk/max/90kDICT32px"
root = "/media/holaverse/aa0e6097-faa0-4d13-810c-db45d9f3bda8/holaverse/work/00ocr/crnn_data/fine_data"


def create_dataset_from_dir(root):
    img_names = os.listdir(root)
    img_paths = []
    for img_name in tqdm(img_names, desc="read dir:"):
        img_name = img_name.rstrip().strip()
        img_path = root + "/" + img_name
        if osp.exists(img_path):
            img_paths.append(img_path)
    labels = [img_path.split("/")[-1].split("_")[-2] for img_path in tqdm(img_paths, desc="generator label:")]
    return img_paths, labels


def create_dataset_from_file(root, file_path):
    with open(file_path, "r") as f:
        readlines = f.readlines()

    img_paths = []
    for img_name in tqdm(readlines, desc="read dir:"):
        img_name = img_name.rstrip().strip()
        img_name = img_name.split(" ")[0]
        img_path = root + "/" + img_name
        # if osp.exists(img_path):
        img_paths.append(img_path)
    img_paths = img_paths[:1000000]
    labels = [img_path.split("/")[-1].split("_")[-2] for img_path in tqdm(img_paths, desc="generator label:")]
    return img_paths, labels


def load_dataset(root):
    img_paths_tensor, labels = create_dataset_from_file(root, root + "/annotation_train.txt")
    print(len(labels))
    labels = [label for label in labels]

    processed_labels = [preprocess_label(label) for label in tqdm(labels, desc="process label:")]

    label_lang = LanguageIndex(label for label in processed_labels)

    labels_tensor = [[label_lang.word2idx[s] for s in label.split(' ')] for label in processed_labels]

    label_max_len = max_length(labels_tensor)

    labels_tensor = tf.keras.preprocessing.sequence.pad_sequences(labels_tensor, maxlen=label_max_len, padding='post')

    return img_paths_tensor, labels_tensor, labels, label_lang, label_max_len


img_paths_tensor, labels_tensor, labels, label_lang, label_max_len = load_dataset(root)

BUFFER_SIZE = len(img_paths_tensor)
BATCH_SIZE = 64
N_BATCH = BUFFER_SIZE // BATCH_SIZE
embedding_dim = 256
units = 1024

vocab_size = len(label_lang.word2idx)


def map_func(img_path_tensor, label_tensor, label):
    imread = cv2.imread(img_path_tensor.decode('utf-8'), cv2.IMREAD_GRAYSCALE)
    if imread is None:
        print(img_path_tensor.decode('utf-8'))
    imread = resize_image(imread, 100, 32)
    imread = np.expand_dims(imread, axis=-1)
    imread = np.array(imread, np.float32)
    return imread, label_tensor, label


dataset = tf.data.Dataset.from_tensor_slices((img_paths_tensor, labels_tensor, labels)) \
    .map(lambda item1, item2, item3: tf.py_func(map_func, [item1, item2, item3], [tf.float32, tf.int32, tf.string]),
         num_parallel_calls=8) \
    .shuffle(1000, reshuffle_each_iteration=True)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

encoder = Encoder(units, BATCH_SIZE)
decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

EPOCHS = 100000

global_step = tf.train.get_or_create_global_step()

logdir = "./logs/"
writer = tf.contrib.summary.create_file_writer(logdir)
writer.set_as_default()

with tf.contrib.summary.record_summaries_every_n_global_steps(1):
    for epoch in range(EPOCHS):
        start = time.time()

        total_loss = 0

        for (batch, (inp, targ, ground_truths)) in enumerate(dataset):
            loss = 0
            global_step.assign_add(1)

            results = np.zeros((BATCH_SIZE, targ.shape[1] - 1), np.int32)

            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp)

                dec_hidden = enc_hidden

                dec_input = tf.expand_dims([label_lang.word2idx['<start>']] * BATCH_SIZE, 1)

                # Teacher forcing - feeding the target as the next input
                for t in range(1, targ.shape[1]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

                    predicted_id = tf.argmax(predictions, axis=-1).numpy()

                    results[:, t - 1] = predicted_id

                    loss += loss_function(targ[:, t], predictions)

                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = (loss / int(targ.shape[1]))

            total_loss += batch_loss

            variables = encoder.variables + decoder.variables

            gradients = tape.gradient(loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))

            preds = [process_result(result, label_lang) for result in results]

            ground_truths = [l.numpy().decode() for l in ground_truths]

            acc = compute_accuracy(ground_truths, preds)

            tf.contrib.summary.scalar('loss', batch_loss)

            if batch % 1 == 0:
                print('Epoch {} Batch {}/{} Loss {:.4f} Mean Loss {:.4f} acc {:f}'.format(epoch + 1, batch, N_BATCH,
                                                                                          batch_loss.numpy(),
                                                                                          total_loss / (batch + 1),
                                                                                          acc))
            if batch % 10 == 0:
                for i in range(5):
                    print("real:{:s}  pred:{:s} acc:{:f}".format(ground_truths[i], preds[i],
                                                                 compute_accuracy([ground_truths[i]], [preds[i]])))

        checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
