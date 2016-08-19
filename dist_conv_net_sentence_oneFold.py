"""
Convolutional Neural Network for Relation Extraction

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import os
warnings.filterwarnings("ignore")   

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)
    
def inspect_inputs(i, node, fn):
    print i, node, "input(s) value(s):", [input[0] for input in fn.inputs],

def inspect_outputs(i, node, fn):
    print "output(s) value(s):", [output[0] for output in fn.outputs]

def writeInfo(f, info):
    old_stdout = sys.stdout
    sys.stdout = f
    print info
    f.flush()
    os.fsync(f.fileno())
    sys.stdout = old_stdout
    print info

def detect_nan(i, node, fn):
    for output in fn.outputs:
        if numpy.isnan(output[0]).any():
            print '*** NaN detected ***'
            theano.printing.debugprint(node)
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            break
       
def train_conv_net(datasets,
                   U,
                   D1,
                   D2,
                   logger,
                   word_vectors,
                   dist_dim=50,
                   img_w=300, 
                   filter_hs=[3,4,5],
                   hidden_units=[100,7], 
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25, 
                   batch_size=50, 
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0])-1
    img_w += 2*dist_dim
    added_dim = 2*dist_dim
    filter_w = img_w
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("epochs",n_epochs),
                  ("dropout", dropout_rate), ("batch_size",batch_size), ("learn_decay",lr_decay),
                  ("conv_non_linear", conv_non_linear), ("sqr_norm_lim",sqr_norm_lim), ("shuffle_batch",shuffle_batch),
                  ("filter_hs",filter_hs), ("non_static", non_static), ("word_vectors", word_vectors), ("hidden_units",hidden_units)]
    writeInfo(logger, parameters)   
    
    #define model architecture
    writeInfo(logger, 'constructing graph ...')
    index = T.lscalar()
    x = T.matrix('x')
    dist1 = T.matrix('dist1')
    dist2 = T.matrix('dist2')
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words") #consider to add borrow=True
    Distance1 = theano.shared(value = D1, name = "Distance1") #consider to add borrow=True
    Distance2 = theano.shared(value = D2, name = "Distance2") #consider to add borrow=True
    zero_vec_tensor = T.vector()
    zero_vec_tensor_dist1 = T.vector()
    zero_vec_tensor_dist2 = T.vector()
    zero_vec = np.zeros(img_w-added_dim,dtype='float32')
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))])
    zero_vect_dist = np.zeros(dist_dim,dtype='float32')
    set_zero_dist1 = theano.function([zero_vec_tensor_dist1], updates=[(Distance1, T.set_subtensor(Distance1[0,:], zero_vec_tensor_dist1))])
    set_zero_dist2 = theano.function([zero_vec_tensor_dist2], updates=[(Distance2, T.set_subtensor(Distance2[0,:], zero_vec_tensor_dist2))])
    
    writeInfo(logger, 'building layer0_input ...')
    layer0_input1 = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))
    layer0_input2 = Distance1[T.cast(dist1.flatten(),dtype="int32")].reshape((dist1.shape[0],1,dist1.shape[1],Distance1.shape[1]))
    layer0_input3 = Distance2[T.cast(dist2.flatten(),dtype="int32")].reshape((dist2.shape[0],1,dist2.shape[1],Distance2.shape[1]))
    
    
    layer0_input = T.cast(T.concatenate([layer0_input1, layer0_input2, layer0_input3], axis=3), dtype=theano.config.floatX)          
    conv_layers = []
    layer1_inputs = []
    writeInfo(logger, 'adding convolutional layers ...')
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)
    writeInfo(logger, 'dropouting ...')
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    
    #define parameters of the model and update functions using adadelta
    writeInfo(logger, 'creating parameters ...')
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
        params += [Distance1]
        params += [Distance2]
    cost = classifier.negative_log_likelihood(y)
    writeInfo(logger, 'computing cost and gradient ...')
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    writeInfo(logger, 'minibatching data ...')
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])   
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
        
        np.random.seed(3435)
        train_dist1 = np.random.permutation(datasets[1])
        extra_dist1 = train_dist1[:extra_data_num]
        new_dist1=np.append(datasets[1],extra_dist1,axis=0)
        
        np.random.seed(3435)
        train_dist2 = np.random.permutation(datasets[2])
        extra_dist2 = train_dist2[:extra_data_num]
        new_dist2=np.append(datasets[2],extra_dist2,axis=0)
    else:
        new_data = datasets[0]
        new_dist1 = datasets[1]
        new_dist2 = datasets[2]
    #new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = n_batches
    writeInfo(logger, 'generating functions ...')
    test_set_x = datasets[3][:,:img_h]
    test_dist1_x = datasets[4][:,:img_h]
    test_dist2_x = datasets[5][:,:img_h]
    test_set_y = np.asarray(datasets[3][:,-1],"int32")
    train_set = new_data[:,:]
    train_dist1 = new_dist1[:,:]
    train_dist2 = new_dist2[:,:]
    train_set_x, train_dist1_x, train_dist2_x, train_set_y = shared_dataset((train_set[:,:img_h],train_dist1[:,:img_h],train_dist2[:,:img_h],train_set[:,-1])) 
            
    #make theano functions to get train/val/test errors            
    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            dist1: train_dist1_x[index*batch_size:(index+1)*batch_size],
            dist2: train_dist2_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]})#, mode=theano.compile.MonitorMode(
                        #pre_func=inspect_inputs,
                        #post_func=inspect_outputs))     
    test_pred_layers = []
    test_size = test_set_x.shape[0]
    test_layer0_input1 = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    test_layer0_input2 = Distance1[T.cast(dist1.flatten(),dtype="int32")].reshape((test_size,1,img_h,Distance1.shape[1]))
    test_layer0_input3 = Distance2[T.cast(dist2.flatten(),dtype="int32")].reshape((test_size,1,img_h,Distance2.shape[1]))
    test_layer0_input = T.cast(T.concatenate([test_layer0_input1, test_layer0_input2, test_layer0_input3], axis=3), dtype=theano.config.floatX)
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)
    
    zero_for_err = T.zeros_like(y)
    numKey_err = T.sum(T.neq(y, zero_for_err))
    numPred_err = T.sum(T.neq(test_y_pred, zero_for_err))
    predIds_err = test_y_pred.nonzero()
    preds_err = test_y_pred[predIds_err]
    keys_err = y[predIds_err]
    correct_err = T.sum(T.eq(preds_err, keys_err))
    p_err = 100.0 * correct_err / numPred_err
    r_err = 100.0 * correct_err / numKey_err
    f_err = (2.0 * p_err * r_err) / (p_err + r_err)
    test_error = (p_err, r_err, f_err)
    
    test_model_all = theano.function([x,dist1,dist2,y], test_error)   
    
    #start training over mini-batches
    writeInfo(logger, '... training')
    epoch = 0     
    cost_epoch = 0
    test_f1 = (0.0,0.0,0.0)
    while (epoch < n_epochs):        
        epoch = epoch + 1
        writeInfo(logger, ('------in epoch %i(%i batches)-----' % (epoch,n_train_batches)))
        runId = 0
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                runId += 1
                cost_epoch = train_model(minibatch_index)
                if runId % 50 == 0:
                    writeInfo(logger, (runId, ': minibatch_index = ', minibatch_index, ', err = ', cost_epoch))
                set_zero(zero_vec)
                set_zero_dist1(zero_vect_dist)
                set_zero_dist2(zero_vect_dist)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)
                set_zero_dist1(zero_vect_dist)
                set_zero_dist2(zero_vect_dist)
        writeInfo(logger, 'updated parameters!')
        test_f1 = test_model_all(test_set_x,test_dist1_x,test_dist2_x,test_set_y)    
        writeInfo(logger, ('      current test perf ', test_f1))
    return test_f1

def shared_dataset(data_xyzt, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y, data_z, data_t = data_xyzt
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_z = theano.shared(np.asarray(data_z,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_t = theano.shared(np.asarray(data_t,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, shared_y, shared_z, T.cast(shared_t, 'int32')
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to
    
def get_idx_from_sent(sent, pos1, pos2, word_idx_map, max_l=20, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    dist1 = []
    dist2 = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
        dist1.append(0)
        dist2.append(0)
    words = sent.split()
    id = -1
    for word in words:
        id += 1
        word = ' '.join(word.split('_'))
        if word in word_idx_map:
            x.append(word_idx_map[word])
            dist1.append(max_l + id - pos1)
            dist2.append(max_l + id - pos2)
        else:
            print 'unrecognized word in get_idx_from_sent ', word
            exit()
    while len(x) < max_l+2*pad:
        x.append(0)
        dist1.append(0)
        dist2.append(0)
    
    return x, dist1, dist2

def make_idx_data_cv(revs, word_idx_map, cv, max_l=20, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, trainDist1, trainDist2, test, testDist1, testDist2 = [], [], [], [], [], []
    for rev in revs:
        sent, dist1, dist2 = get_idx_from_sent(rev["text"], rev["pos1"], rev["pos2"], word_idx_map, max_l, k, filter_h) 
         
        sent.append(rev["y"])
        dist1.append(0)
        dist2.append(0)
        if rev["fold"]==cv:            
            test.append(sent)
            testDist1.append(dist1)
            testDist2.append(dist2)      
        else:  
            train.append(sent)
            trainDist1.append(dist1)
            trainDist2.append(dist2)
    train = np.array(train,dtype="int")
    trainDist1 = np.array(trainDist1,dtype="int")
    trainDist2 = np.array(trainDist2,dtype="int")
    test = np.array(test,dtype="int")
    testDist1 = np.array(testDist1,dtype="int")
    testDist2 = np.array(testDist2,dtype="int")
    return [train, trainDist1, trainDist2, test, testDist1, testDist2]   
  
   
if __name__=="__main__":
    hidden_units = [150,7] # 50, 100, 150, 200
    filter_h = 5
    filter_hs = [2,3,4,5]
    mode = "static" # static, nonstatic
    word_vectors = "word2vec" # rand, word2vec
    fold = 0 # 0, 1, 2, 3, 4
    
    if mode=="nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="static":
        print "model architecture: CNN-static"
        non_static=False
    else:
        print 'Undefined mode'
        exit()
    
    if word_vectors=="rand":
        print "using: random vectors"
    elif word_vectors=="word2vec":
        print "using: word2vec vectors"
    else:
        print 'Undefined word vectors'
        exit()
        
    execfile("dist_conv_net_classes.py")
    
    filterStr = ''
    for fil in filter_hs:
        filterStr = filterStr + '-' + str(fil)
    filterStr = filterStr[1:]
    log_file = 'h' + str(hidden_units[0]) + '.' + mode + '.' + word_vectors + '.f' + filterStr + '.fold' + str(fold)
    
    logger = open('res/' + log_file,"w")
    writeInfo(logger, 'setting=' + log_file)
    writeInfo(logger, "loading data...")
    x = cPickle.load(open("distnnre.dat","rb"))
    revs, W, W2, D1, D2, word_idx_map, vocab, id2Label = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
    writeInfo(logger, "data loaded!")
    
    if word_vectors=="rand":
        U = W2
        k = 300
    elif word_vectors=="word2vec":
        U = W
        k = 300
    max_l = 20
    #dist_size = 2*max_l - 1
    dist_dim = 50
    #D1 = np.random.uniform(-0.25,0.25,(dist_size+1,dist_dim))
    #D2 = np.random.uniform(-0.25,0.25,(dist_size+1,dist_dim))
    #D1[0] = np.zeros(dist_dim)
    #D2[0] = np.zeros(dist_dim)
    
    writeInfo(logger, ('------------------------------working on fold %i------------------' % fold))
    writeInfo(logger, 'producing data for this fold ...')
    datasets = make_idx_data_cv(revs, word_idx_map, fold, max_l=max_l, k=k, filter_h=filter_h)
    perf = train_conv_net(datasets,
                          U,
                          D1,
                          D2,
                          logger,
                          word_vectors=word_vectors,
                          dist_dim=dist_dim,
                          lr_decay=0.95,
                          filter_hs=filter_hs,
                          conv_non_linear="tanh",
                          hidden_units=hidden_units,
                          shuffle_batch=True,
                          n_epochs=20,
                          sqr_norm_lim=9,
                          non_static=non_static,
                          batch_size=50,
                          dropout_rate=[0.5])
    writeInfo(logger, ("cv: " + str(fold) + ", perf: " + str(perf)))
    writeInfo(logger, 'Done ' + log_file + '!')
    logger.close()