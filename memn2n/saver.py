from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data, load_task_UBT, load_task_single
from sklearn.model_selection import cross_validate as cross_validation
from sklearn.model_selection import train_test_split
from sklearn import metrics
from memn2n import MemN2N
from itertools import chain
from six.moves import range, reduce

import tensorflow as tf
import numpy as np
import pickle

batch_size = 32
embedding_size = 200
hops = 7
max_grad_norm = 40.0

saver = tf.train.import_meta_graph('saved_models/memn2n.ckpt.meta')
#saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with open ('outvocab', 'rb') as fp:
    vocab = pickle.load(fp)

with tf.Session(config = config) as sess:
    #saver.restore(sess,'saved_models/.')
    saver.restore(sess, "saved_models/memn2n.ckpt")
    vocab_size = tf.get_collection('_vocab_size')[0]
    sentence_size = tf.get_collection('_sentence_size')[0]
    memory_size = tf.get_collection('_memory_size')[0]
    print(memory_size)
    print(sentence_size)
    print(vocab_size)

    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, embedding_size, sess, hops,max_grad_norm)

    print("---------------------------------- loading test data ----------------------------------")
    single_data, single_onehot_answer, single_word_answer, single_choices = load_task_single(FLAGS.data_dir, "simple")
    testS, testQ, testA = vectorize_data(single_data, word_idx, sentence_size, memory_size)
    test_labels = np.argmax(testA, axis=1)
    test_preds = model.predict(testS, testQ)
    test_acc = metrics.accuracy_score(test_preds, test_labels)
    #check
    word_idx_switch = {y:x for x,y in word_idx.items()}
    for i in range(len(np.array(test_labels))) :
        print("Ques-",i+1,"----------------------------------")
        print(np.array(single_choices[i]))
        print("Answer : ", word_idx_switch[np.array(test_labels)[i]]," Predic : ", word_idx_switch[np.array(test_preds)[i]])
    print("Testing Accuracy:", test_acc)
