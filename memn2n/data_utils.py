from __future__ import absolute_import

import os
import re
import numpy as np
import sys

def load_task(data_dir, task_id, only_supporting=False):
    '''Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir) 
    files = [os.path.join(data_dir, f) for f in files] 
    s = 'qa{}_'.format(task_id) #ex. qa1_
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data

def load_task_UBT(data_dir, task_name, only_supporting=False):
    '''Load UBT task, data_dir is folder path, task_name is file to open
    Returns data for the task.
    '''
    files = os.listdir(data_dir) 
    files = [os.path.join(data_dir, f) for f in files] 
    #print(files)
    train_file = [f for f in files if task_name in f and 'train' in f][0]
    #print(train_file)
    test_file = [f for f in files if task_name in f and 'test' in f][0]
    train_data = get_stories_UBT(train_file)
    test_data = get_stories_UBT(test_file)
    return train_data, test_data
    #train_data, train_answer = get_stories_UBT(train_file)
    #test_data, test_answer = get_stories_UBT(test_file)
    #return train_data, train_answer, test_data, test_answer

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1) 
        nid = int(nid)
        if nid == 1: 
            story = []
        if '\t' in line: 
            q, a, supporting = line.split('\t') 
            q = tokenize(q)
            #a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [a]
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append('')
        else: # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data

def parse_stories_UBT(lines):
    '''Parse stories provided in the UBT tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    run = 0
    onehot_answer = []
    for line in lines:
        #run = run + 1
        if run == 100:
            break
        #print(type(line))
        #line = line.encode("utf8")
        #line = line.encode(sys.stdin.encoding, "replace").decode(sys.stdin.encoding)
        #line = str(line)
        line = str.lower(line)
        #print(str_line)
        if line != "\n":
            nid, line = line.split(' ', 1) 
            nid = int(nid)
        else : 
            nid = 0
        #print(nid)
        if nid == 1: 
            story = []
        elif nid == 21: # question 
            run = run + 1
            choice = line.split()[-1] #choice
            answer = line.split()[-2] #answer
            #question = line.replace(choice, "") #answer also has choice in it
            question = line.replace(answer, "")

            question = question.replace('\'', '') #do n't -> do nt
            question = question.replace('-', '') #lunber-room -> lumberroom
            question = question.replace('.', '') #2.png -> 2png
            question = re.sub(r'[^a-zA-Z0-9]', ' ', question)
            question = tokenize(question)
            #print(question)

            choice = choice.replace('\'', '') #do n't -> do nt
            choice = choice.replace('-', '') #lunber-room -> lumberroom
            choice = choice.replace('.', '') 
            choice = choice.replace(',', '') #1000,000
            choice = choice.replace('|', ' ') 
            choice = tokenize(choice)
            #print("choice", choice)

            answer = answer.replace('\'', '') #do n't -> do nt
            answer = answer.replace('-', '') #lunber-room -> lumberroom
            answer = answer.replace('.', '') #2.png -> 2png
            answer = re.sub(r'[^a-zA-Z0-9]', ' ', answer)
            answer = tokenize(answer)
            #print("answer", answer)

	    #try to convert wanted output answer format
            #choice_idx = dict((c, i + 1) for i, c in enumerate(choice))
            #answer_onehot = np.zeros(10) 
            #print(run)
            #for a in answer:
            #    answer_onehot[choice_idx[a]-1] = 1
            #if len(answer_onehot) != 10 :
            #    print("choice", choice)
            #    print("answer", answer)
            #    print("answer_onehot", answer_onehot) 

	    #onehot_answer.append((answer_onehot))
            data.append((story, question, answer))

        elif nid != 0: 
            line = line.replace('\'', '') #do n't -> do nt
            line = re.sub(r'[^a-zA-Z0-9]', ' ', line)
            sent = tokenize(line)
            #print(sent)
            story.append(sent)
        #else: # regular sentence
    #return data, np.array(onehot_answer)
    return data

def parse_stories_single(lines):
    '''Parse stories provided in the UBT tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    onehot_answer = []
    choices = []
    word_answer = []
    for line in lines:
        #line = line.decode("utf-8")
        #line = line.encode(sys.stdin.encoding, "replace").decode(sys.stdin.encoding)
        line = str.lower(line)
        if line != "\n":
            nid, line = line.split(' ', 1) #(ex. "8" Sandra journeyed to the bathroom.)
            nid = int(nid)
        else : 
            nid = 0
        #print(nid)
        if nid == 1: 
            story = []
        elif nid == 21: # question 
            choice = line.split()[-1] #choice
            answer = line.split()[-2] #answer
            #question = line.replace(choice, "") #answer also has choice in it
            question = line.replace(answer, "")

            question = question.replace('\'', '') #do n't -> do nt
            question = question.replace('-', '') #lunber-room -> lumberroom
            question = question.replace('.', '') #2.png -> 2png
            question = re.sub(r'[^a-zA-Z0-9]', ' ', question)
            question = tokenize(question)
            #print(question)

            choice = choice.replace('\'', '') #do n't -> do nt
            choice = choice.replace('-', '') #lunber-room -> lumberroom
            choice = choice.replace('.', '') 
            choice = choice.replace(',', '') #1000,000 -> 1000000
            choice = choice.replace('|', ' ') 
            choice = tokenize(choice)
            #print("choice", choice)

            answer = answer.replace('\'', '') #do n't -> do nt
            answer = answer.replace('-', '') #lunber-room -> lumberroom
            answer = answer.replace('.', '') #2.png -> 2png
            answer = re.sub(r'[^a-zA-Z0-9]', ' ', answer)
            answer = tokenize(answer)
            #print("answer", answer)

            #try to convert wanted output answer format
            choice_idx = dict((c, i + 1) for i, c in enumerate(choice))
            answer_onehot = np.zeros(10) 
            for a in answer:
                answer_onehot[choice_idx[a]-1] = 1
            if len(answer_onehot) != 10 :
                print("choice", choice)
                print("answer", answer)
                print("answer_onehot", answer_onehot) 
            #print(answer_onehot)
            
            onehot_answer.append((answer_onehot))
            data.append((story, question, answer))
            word_answer.append(answer)
            choices.append(choice)

        elif nid != 0: 
            line = line.replace('\'', '') #do n't -> do nt
            line = re.sub(r'[^a-zA-Z0-9]', ' ', line)
            sent = tokenize(line)
            story.append(sent)
    return data, np.array(onehot_answer), word_answer, choices
def get_stories_single(f):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f, 'rb') as f:
        return parse_stories_single(f.readlines())
def load_task_single(data_dir, task_name, only_supporting=False):
    '''Load UBT task, data_dir is folder path, task_name is file to open
    Returns data for the task.
    '''
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files] 
    train_file = [f for f in files if task_name in f and 'test' in f][0]
    data, onehot_answer, word_answer, choices = get_stories_single(train_file)
    return data, onehot_answer, word_answer, choices


def get_stories_UBT(f):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f, 'rb') as f:
        return parse_stories_UBT(f.readlines())

def get_stories(f, only_supporting=False):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)

def vectorize_data(data, word_idx, sentence_size, memory_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        ss = []
	#sc = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            #ss.append([word_idx[w] for w in sentence] + [0] * ls)
            #print("ss", ss)
  	    #value = d.get(key, "empty")
	    st = []
            for w in sentence :
		value = word_idx.get(w, 0)
                #if word_idx[w] :
                st.append(value)#sentence
		#else :
		#st.append(0)
	    ss.append(st + [0] * ls)
            #print("sc",sc)
        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # Make the last word of each sentence the time 'word' which 
        # corresponds to vector of lookup table
        for i in range(len(ss)):
            ss[i][-1] = len(word_idx) - memory_size - i + len(ss)

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A)
