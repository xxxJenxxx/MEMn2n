"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data, load_task_UBT, load_task_single
from sklearn.model_selection import cross_validate as cross_validation
from sklearn.model_selection import train_test_split
from sklearn import metrics
#from sklearn import cross_validation, metrics
from memn2n import MemN2N
from itertools import chain
from six.moves import range, reduce
import pickle

import tensorflow as tf
import os
import numpy as np

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for SGD.")
tf.flags.DEFINE_float("anneal_rate", 25, "Number of epochs between halving the learnign rate.")
tf.flags.DEFINE_float("anneal_stop_epoch", 100, "Epoch number to end annealed lr schedule.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 7, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs",100, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 200, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 200, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 3, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
#tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("data_dir", "data/CBTest/data/", "Directory containing CBT tasks")
FLAGS = tf.flags.FLAGS

#print("Started Task:", FLAGS.task_id)

# task data
#train, test = load_task(FLAGS.data_dir, FLAGS.task_id)
#train, test = load_task_UBT(FLAGS.data_dir, "NE")
train, test = load_task_UBT(FLAGS.data_dir, "NE")
#train, train_ans, test, test_ans = load_task_UBT(FLAGS.data_dir, "NE")
#-------------------------------------------------------------------------------------------------
data = train + test

#data format : (1000,3) 0->s sentences, 1->q question, 2->a answer
#print(np.asarray(train)[1,0])

vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
with open('outvocab', 'wb') as fp:
    pickle.dump(vocab, fp)

word_idx = dict((c, i + 1) for i, c in enumerate(vocab)) 

max_story_size = max(map(len, (s for s, _, _ in data)))
mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
query_size = max(map(len, (q for _, q, _ in data)))
memory_size = min(FLAGS.memory_size, max_story_size)

# Add time words/indexes  ?? why add time ?? 'time2': 'time2', 'ti...
for i in range(memory_size):
    word_idx['time{}'.format(i+1)] = 'time{}'.format(i+1)

#print(len(word_idx))
vocab_size = len(word_idx) + 1 # +1 for nil word
sentence_size = max(query_size, sentence_size) # for the position
sentence_size += 1  # +1 for time words

print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)

# train/validation/test sets
S, Q, A = vectorize_data(train, word_idx, sentence_size, memory_size) #use word_idx to do encoding
#print(np.asarray(A).shape)
#print(np.asarray(Q).shape)
#print(np.asarray(S).shape)
trainS, valS, trainQ, valQ, trainA, valA = train_test_split(S, Q, A, test_size=.1, random_state=FLAGS.random_state)
#trainS, valS, trainQ, valQ, trainA, valA = train_test_split(S, Q, train_ans, test_size=.1, random_state=FLAGS.random_state)
testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)

#print(testS[0])

print("Training set shape", trainS.shape)

# params
n_train = trainS.shape[0]
n_test = testS.shape[0]
n_val = valS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

train_labels = np.argmax(trainA, axis=1)
test_labels = np.argmax(testA, axis=1)
#test_labels = np.argmax(test_ans, axis=1)
val_labels = np.argmax(valA, axis=1)

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size

batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config = config) as sess:

    model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, FLAGS.embedding_size, sess,
                   FLAGS.hops, FLAGS.max_grad_norm)
    for t in range(1, FLAGS.epochs+1):
    #for t in range():
        # Stepped learning rate
        if t - 1 <= FLAGS.anneal_stop_epoch:
            anneal = 2.0 ** ((t - 1) // FLAGS.anneal_rate)
        else:
            anneal = 2.0 ** (FLAGS.anneal_stop_epoch // FLAGS.anneal_rate)
        lr = FLAGS.learning_rate / anneal

        np.random.shuffle(batches)
        total_cost = 0.0
        for start, end in batches:
            s = trainS[start:end]
            q = trainQ[start:end]
            a = trainA[start:end]
            cost_t = model.batch_fit(s, q, a, lr)
            total_cost += cost_t

        if t % FLAGS.evaluation_interval == 0:
            train_preds = []
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                s = trainS[start:end]
                q = trainQ[start:end]
                pred = model.predict(s, q)
                train_preds += list(pred)

            val_preds = model.predict(valS, valQ)
            train_acc = metrics.accuracy_score(np.array(train_preds), train_labels) #calculate cost
	    #print("train_labels")
            #print(np.array(train_labels))
	    #print("train_preds")
            #print(np.array(train_preds))
            val_acc = metrics.accuracy_score(val_preds, val_labels)

            print('-----------------------')
            print('Epoch', t)
            print('Total Cost:', total_cost)
            print('Training Accuracy:', train_acc)
            print('Validation Accuracy:', val_acc)
            print('-----------------------')

    """test_preds = model.predict(testS, testQ)
    test_acc = metrics.accuracy_score(test_preds, test_labels)
    print("Testing Accuracy:", test_acc)"""
    """#-------------------------------------------------------------------------------------------------
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
        print(np.array(single_choices)[i])
        print("Answer : ", word_idx_switch[np.array(test_labels)[i]]," Predic : ", word_idx_switch[np.array(test_preds)[i]])
    print("Testing Accuracy:", test_acc)
    #-------------------------------------------------------------------------------------------------
    """
    #save model
    modelSaver = tf.train.Saver()
    modelSaver.save(sess, "saved_models/memn2n.ckpt")

#with open ('outvocab', 'rb') as fp:
#    v = pickle.load(fp)

#print(v)
