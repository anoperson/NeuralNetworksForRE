import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

#thien's version
window = 20

def build_data_cv(data_folder, data_file):
    """
    Loads data.
    """
    labelDict = {'classLabel=Other':0}
    vocab = defaultdict(float)
    revs = []
    file2Fold = loadFoldMap(data_folder)
    fold = 0
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') and line.endswith('#'):
                line = line[1:len(line)-1]
                if not file2Fold.has_key(line):
                    print 'cannot find fold for file: ', line
                    exit()
                fold = file2Fold[line]
                continue
            
            relId, features, _, classLabel, type1, subtype1, pos1, type2, subtype2, pos2, sentence = parseLine(line)
            if not labelDict.has_key(classLabel):
                labelDict[classLabel] = len(labelDict)
                print 'label: ', classLabel, ' --> id = ', labelDict[classLabel]
                if classLabel == 'classLabel=Other#':
                    print '-------Wrong: ', line
            classId = labelDict[classLabel]
            
            sentence, pos1, pos2 = fitSentenceToWindow(sentence=sentence, pos1=pos1, pos2 = pos2, window=window)
            
            #print 'sent: ', sentence, ' len = ', len(sentence.split()), ' poss: ', pos1, pos2, classLabel
            
            words = set(sentence.split())
            for word in words:
                word = ' '.join(word.split('_'))
                vocab[word] += 1
            
            datum = {"id": relId,
                     "y":classId,
                     "text": sentence,
                     "pos1": pos1,
                     "pos2": pos2,
                     "type1": type1,
                     "subtype1": subtype1,
                     "type2": type2,
                     "subtype2": subtype2,
                     "fold": fold}
            revs.append(datum)
    return revs, vocab, labelDict

def parseLine(line):
    relId = line[0:line.find(' ')]
    line = line[(line.find(' ')+1):]
    rel = line[0:line.rfind('###@@@')]
    sentInfo = line[(line.rfind('###@@@')+6):].split('\t')
    
    classLabel = rel[(rel.rfind(' ')+1):]
    rel = rel[0:rel.rfind(' ')]
    detectorLabel = rel[(rel.rfind(' ')+1):]
    features = rel[0:rel.rfind(' ')]
    
    sentence = sentInfo[0].lower()
    _, type1, subtype1, pos1 = tuple(sentInfo[1].split('#'))
    _, type2, subtype2, pos2 = tuple(sentInfo[2].split('#'))
    
    return relId, features, detectorLabel, classLabel, type1, subtype1, int(pos1), type2, subtype2, int(pos2), sentence
    
def loadFoldMap(data_folder):
    file2Fold = {}
    count = 0
    for fold in range(len(data_folder)):
        with open(data_folder[fold], "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    file2Fold[line] = fold
                    count += 1
    print 'Loaded ', count, ' files!'
    return file2Fold

def fitSentenceToWindow(sentence, pos1, pos2, window=20):
    if (abs(pos1 - pos2)+1) > window:
        print 'Encounter a sentence with two far arguments: ', sentence, pos1, pos2
        exit()
    lower = ((pos1 + pos2 + window) / 2) - window + 1
    npos1 = pos1 - lower
    npos2 = pos2 - lower
    words = sentence.split()
    nsent = ''
    for i in range(window):
        id = lower + i
        nsent += ((words[id] + ' ') if (0 <= id < len(words)) else '###### ')
    
    return nsent.strip(), npos1, npos2
    
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k))            
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

if __name__=="__main__":
    w2v_file = sys.argv[1]
    #data_file = 'dataTest'
    data_file = sys.argv[2]
    basePath = '../relationExtractor/crossValiFilelists/ace2005_clean_test.cv'    
    data_folder = []
    for i in range(5):
        data_folder += [basePath + str(i+1)]  
    print "loading data...\n"
    revs, vocab, labelDict = build_data_cv(data_folder, data_file)
    max_l = np.max(abs(pd.DataFrame(revs)["pos1"] - pd.DataFrame(revs)["pos2"]))
    print "max sentence length: " + str(max_l)
    print "data loaded!"
    print "number of relation instances: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    mmax_l = 20
    dist_size = 2*mmax_l - 1
    dist_dim = 50
    D1 = np.random.uniform(-0.25,0.25,(dist_size+1,dist_dim))
    D2 = np.random.uniform(-0.25,0.25,(dist_size+1,dist_dim))
    D1[0] = np.zeros(dist_dim)
    D2[0] = np.zeros(dist_dim)
    cPickle.dump([revs, W, W2, D1, D2, word_idx_map, vocab, labelDict], open("distnnre.dat", "wb"))
    print "dataset created!"   
