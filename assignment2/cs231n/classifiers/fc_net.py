import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it

  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params['W1']=np.random.normal(0, weight_scale,(input_dim,hidden_dim))
    self.params['W2']=np.random.normal(0, weight_scale,(hidden_dim,num_classes))
    self.params['b1']=np.zeros((1,hidden_dim))
    self.params['b2']=np.zeros((1,num_classes))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    N=X.shape[0]
    # print(X.shape)
    X_compressed = X.reshape((N,np.product(X.shape)/N))
    a1=np.maximum(0,X_compressed.dot(self.params['W1']) + self.params['b1'])
    scores= a1.dot(self.params['W2'])+self.params['b2']
    # W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
    # # Forward into first layer
    # hidden_layer, cache_hidden_layer = affine_relu_forward(X, W1, b1)
    # # Forward into second layer
    # scores, cache_scores = affine_forward(hidden_layer, W2, b2)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    N=X.shape[0]
    num_train=N
    reg=self.reg
    W2=self.params['W2']
    W1=self.params['W1']
    exp_scores=np.exp(scores)
    # prob=exp_scores/np.sum(exp_scores,axis=1)
    exp_sum=np.sum(exp_scores,axis=1)
    exp_sum=exp_sum.reshape(len(exp_sum),1)
    prob=np.divide(exp_scores,exp_sum)
    y_raw=prob[np.arange(N),y]
    sum_yraw=-np.log(y_raw)
    loss=np.sum(sum_yraw)

    dScores = prob
    dScores[np.arange(num_train),y]-=1
    dScores/=num_train

    grads['W2']= np.dot(a1.T,dScores) + reg * self.params['W2']
    grads['b2']= np.sum(dScores,axis=0)
    d_hidden_layer= np.dot(dScores ,self.params['W2'].T)
    d_hidden_layer[a1<=0]=0
    grads['W1'] = np.dot(X_compressed.T,d_hidden_layer) + reg * self.params['W1']
    grads['b1'] = np.sum(d_hidden_layer,axis=0)

    loss/=N
    loss+= reg*0.5*(np.sum(W1*W1) + np.sum(W2*W2))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    # all_dims= [input_dim] + hidden_dims + [num_classes]
    # # print(all_dims)
    # # print(self.num_layers)
    # for i in range(self.num_layers ):
    #     # print(i+1)
    #     self.params['b%d'%(i+1)]=np.zeros(all_dims[i+1])
    #     self.params['W%d'%(i+1)]=np.random.normal(0,weight_scale,[all_dims[i],all_dims[i+1]])
    #     # print(self.params['b%d'%(i+1)].shape)
    #     # print(self.params['W%d'%(i+1)].shape)
    #     # print(X.shape)
    # for i in range(self.num_layers-1)
    #     if self.use_batchnorm:
    #         self.params['beta%d'%(i+1)] = np.zeros([all_dims[i+1]])
    #         self.params['gamma%d'%(i+1)] = np.ones([all_dims[i+1]])
    
    # # print("end")
    
    for i in range(self.num_layers - 1):
        self.params['W' + str(i+1)] = np.random.normal(0, weight_scale, [input_dim, hidden_dims[i]])
        self.params['b' + str(i+1)] = np.zeros([hidden_dims[i]])

        if self.use_batchnorm:
            self.params['beta' + str(i+1)] = np.zeros([hidden_dims[i]])
            self.params['gamma' + str(i+1)] = np.ones([hidden_dims[i]])

        input_dim = hidden_dims[i]  # Set the input dim of next layer to be output dim of current layer.

    # Initialise the weights and biases for final FC layer
    self.params['W' + str(self.num_layers)] = np.random.normal(0, weight_scale, [input_dim, num_classes])
    self.params['b' + str(self.num_layers)] = np.zeros([num_classes])



    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  # def loss(self, X, y=None):
 #    """
 #    Compute loss and gradient for the fully-connected net.

 #    Input / output: Same as TwoLayerNet above.
 #    """
 #    X = X.astype(self.dtype)
 #    mode = 'test' if y is None else 'train'

 #    # Set train/test mode for batchnorm params and dropout param since they
 #    # behave differently during training and testing.
 #    if self.dropout_param is not None:
 #      self.dropout_param['mode'] = mode   
 #    if self.use_batchnorm:
 #      for bn_param in self.bn_params:
 #        bn_param[mode] = mode

 #    scores = None
 #    ############################################################################
 #    # TODO: Implement the forward pass for the fully-connected net, computing  #
 #    # the class scores for X and storing them in the scores variable.          #
 #    #                                                                          #
 #    # When using dropout, you'll need to pass self.dropout_param to each       #
 #    # dropout forward pass.                                                    #
 #    #                                                                          #
 #    # When using batch normalization, you'll need to pass self.bn_params[0] to #
 #    # the forward pass for the first batch normalization layer, pass           #
 #    # self.bn_params[1] to the forward pass for the second batch normalization #
 #    # layer, etc.                                                              #
 #    ############################################################################
 #    # a = {}
 #    # a['a1'] = X_compressed.dot(self.params['W1']) + self.params['b1']
 #    # for i in range(self.num_layers-1):
 #    #     a['a%d'%(i+2)] = a['a%d'%(i+1)].dot(self.params['W%d'%(i+1)]) + self.params['b%d'%(i+2)]
 #    aff_for_cache ={}
 #    relu_cache ={}
 #    bn_cache = {}
 #    dropout_cache ={}
 #    N=X.shape[0]
 #    X_compressed = np.reshape(X, [N, -1])
   
 #    for i in range(self.num_layers-1):
 #        # print(self.params['b%d'%(i+1)].shape)
 #        # print(self.params['W%d'%(i+1)].shape)
 #        # print(X.shape)
 #        # print("ok")
 #        affine_forward_act,aff_for_cache[str(i+1)] = affine_forward(X , self.params['W%d'%(i+1)] , self.params['b%d'%(i+1)])
 #        if self.use_batchnorm:
 #            batchnorm_fw_act , bn_cache[str(i+1)] = batchnorm_forward(affine_forward_act , self.params['gamma%d'%(i+1)] , self.params['beta%d'%(i+1)] , self.bn_params[i])
 #            relu_act , relu_cache[str(i+1)] = relu_forward(batchnorm_fw_act)
 #        else:
 #            relu_act , relu_cache[str(i+1)] = relu_forward(affine_forward_act)
 #        final_act = relu_act
 #        if self.use_dropout:
 #            dropout_act , dropout_cache[str(i+1)] = dropout_forward(relu_act , self.dropout_param)
 #            final_act = dropout_act
 #        X = final_act.copy()

 #    # final output layer without relu
 #    # print(self.params['b%d'%(self.num_layers)].shape)
 #    # print(self.params['W%d'%(self.num_layers)].shape)
 #    # print(X.shape)
 #    scores , final_cache = affine_forward(X, self.params['W%d'%(self.num_layers)] , self.params['b%d'%(self.num_layers)])
 # # bn_act, bn_cache[str(i+1)] = batchnorm_forward(fc_act, self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], self.bn_params[i])

 #    ############################################################################
 #    #                             END OF YOUR CODE                             #
 #    ############################################################################

 #    # If test mode return early
 #    if mode == 'test':
 #      return scores

 #    loss, grads = 0.0, {}
 #    ############################################################################
 #    # TODO: Implement the backward pass for the fully-connected net. Store the #
 #    # loss in the loss variable and gradients in the grads dictionary. Compute #
 #    # data loss using softmax, and make sure that grads[k] holds the gradients #
 #    # for self.params[k]. Don't forget to add L2 regularization!               #
 #    #                                                                          #
 #    # When using batch normalization, you don't need to regularize the scale   #
 #    # and shift parameters.                                                    #
 #    #                                                                          #
 #    # NOTE: To ensure that your implementation matches ours and you pass the   #
 #    # automated tests, make sure that your L2 regularization includes a factor #
 #    # of 0.5 to simplify the expression for the gradient.                      #
 #    ############################################################################
    
 #    # exp_scores = np.exp(scores)
 #    # prob = exp_scores / np.sum(exp_scores ,axis =1)
 #    # y_raw = prob[np.arange(N) , y]
 #    # prob_log = - np.log(y_raw)
 #    # loss = np.sum(prob_log)
 #    # loss/=N
 #    loss_without_reg ,d_final = softmax_loss(scores , y)

 #    # for the last layer
 #    dx_last , dw_last ,db_last = affine_backward(d_final , final_cache)

 #    # grads of the last layer

 #    grads['W'+str(self.num_layers)] = dw_last + self.reg * self.params['W'+str(self.num_layers)]
 #    grads['b'+str(self.num_layers)] = db_last

 #    # Back-prop
 #    reg_loss_sum= 0

 #    for i in range(self.num_layers-1, 0,-1):
 #        # print(i)
 #        if self.use_dropout:
 #            dx_last = dropout_backward(dx_last , dropout_cache[str(i)])

 #        d_relu = relu_backward(dx_last ,relu_cache[str(i)])

 #        if self.use_batchnorm:
 #            d_batchnorm , d_gamma , d_beta = batchnorm_backward(d_relu , bn_cache[str(i)])
 #            dx_last, dw_last, db_last = affine_backward(d_batchnorm, aff_for_cache[str(i)])
 #            grads['beta' + str(i)] = d_beta
 #            grads['gamma' + str(i)] = d_gamma
 #        else:
 #            dx_last, dw_last, db_last = affine_backward(d_relu, aff_for_cache[str(i)])

 #        # store grads and final loss
 #        reg_loss_sum += np.sum(self.params['W%d'%(i+1)] * self.params['W%d'%(i+1)])
 #        grads['W%d'%(i)] = dw_last + self.reg*self.params['W%d'%(i)]
 #        grads['b'+str(i)] = db_last
 #        # print("#######db_last - %d #######"%i)
 #        # print(grads['b'+str(i)])
 #        # print("#######dw_last### @END@ ####")
 #        # if grads['b'+str(i)].all()==0:
 #        #     print("@@@@@@@@@@@@@@")

 #    reg_loss_sum+= np.sum(np.square(self.params['W'+str(self.num_layers)]))
 #    loss+=0.5 * self.reg *(reg_loss_sum)
 #    ############################################################################
 #    #                             END OF YOUR CODE                             #
 #    ############################################################################
 #    # print("loss n grads")
 #    # print(loss)
 #    # print(grads)
 #    return loss, grads
  def loss(self, X, y=None):
      """
      Compute loss and gradient for the fully-connected net.
      Input / output: Same as TwoLayerNet above.
      """
      X = X.astype(self.dtype)
      mode = 'test' if y is None else 'train'

      # Set train/test mode for batchnorm params and dropout param since they
      # behave differently during training and testing.
      if self.use_dropout:
          self.dropout_param['mode'] = mode
      if self.use_batchnorm:
          for bn_param in self.bn_params:
              bn_param['mode'] = mode

      scores = None
      ############################################################################
      # TODO: Implement the forward pass for the fully-connected net, computing  #
      # the class scores for X and storing them in the scores variable.          #
      #                                                                          #
      # When using dropout, you'll need to pass self.dropout_param to each       #
      # dropout forward pass.                                                    #
      #                                                                          #
      # When using batch normalization, you'll need to pass self.bn_params[0] to #
      # the forward pass for the first batch normalization layer, pass           #
      # self.bn_params[1] to the forward pass for the second batch normalization #
      # layer, etc.                                                              #
      ############################################################################

      hidden_num = self.num_layers - 1
      scores = X
      cache_history = []
      L2reg = 0
      for i in range(hidden_num):
          if self.use_batchnorm:
              scores, cache = affine_bn_relu_forward(scores,
                                                    self.params['W%d' % (i + 1)],
                                                    self.params['b%d' % (i + 1)],
                                                    self.params['gamma%d' % (i + 1)],
                                                    self.params['beta%d' % (i + 1)],
                                                    self.bn_params[i])
          else:
              scores, cache = affine_relu_forward(scores, self.params['W%d' % (i + 1)],
                                                          self.params['b%d' % (i + 1)])
          cache_history.append(cache)
          if self.use_dropout:
              scores, cache = dropout_forward(scores, self.dropout_param)
              cache_history.append(cache)
          L2reg += np.sum(self.params['W%d' % (i + 1)] ** 2)
      i += 1
      scores, cache = affine_forward(scores, self.params['W%d' % (i + 1)],
                                             self.params['b%d' % (i + 1)])
      cache_history.append(cache)
      L2reg += np.sum(self.params['W%d' % (i + 1)] ** 2)
      L2reg *= 0.5 * self.reg

      ############################################################################
      #                             END OF YOUR CODE                             #
      ############################################################################

      # If test mode return early
      if mode == 'test':
          return scores

      loss, grads = 0.0, {}
      ############################################################################
      # TODO: Implement the backward pass for the fully-connected net. Store the #
      # loss in the loss variable and gradients in the grads dictionary. Compute #
      # data loss using softmax, and make sure that grads[k] holds the gradients #
      # for self.params[k]. Don't forget to add L2 regularization!               #
      #                                                                          #
      # When using batch normalization, you don't need to regularize the scale   #
      # and shift parameters.                                                    #
      #                                                                          #
      # NOTE: To ensure that your implementation matches ours and you pass the   #
      # automated tests, make sure that your L2 regularization includes a factor #
      # of 0.5 to simplify the expression for the gradient.                      #
      ############################################################################

      loss, dout = softmax_loss(scores, y)
      loss += L2reg

      dout, grads['W%d' % (i + 1)], grads['b%d' % (i + 1)] = affine_backward(dout, cache_history.pop())
      grads['W%d' % (i + 1)] += self.reg * self.params['W%d' % (i + 1)]
      i -= 1
      while i >= 0:
          if self.use_dropout:
              dout = dropout_backward(dout, cache_history.pop())
          if self.use_batchnorm:
              dout, grads['W%d' % (i + 1)], grads['b%d' % (i + 1)], grads['gamma%d' % (i + 1)], grads['beta%d' % (i + 1)] = affine_bn_relu_backward(dout, cache_history.pop())
          else:
              dout, grads['W%d' % (i + 1)], grads['b%d' % (i + 1)] = affine_relu_backward(dout, cache_history.pop())
          grads['W%d' % (i + 1)] += self.reg * self.params['W%d' % (i + 1)]
          i -= 1

      ############################################################################
      #                             END OF YOUR CODE                             #
      ############################################################################

      return loss, grads


# Batch-Normalization Layer Utilities
def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that performs an affine transform, batch normalization and ReLU
    """
    out1, fc_cache = affine_forward(x, w, b)
    out2, bn_cache = batchnorm_forward(out1, gamma, beta, bn_param)
    out3, relu_cache = relu_forward(out2)
    cache = (fc_cache, bn_cache, relu_cache)
    return out3, cache

def affine_bn_relu_backward(dout, cache):
    """
        Backward pass for the affine-bn-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    d1 = relu_backward(dout, relu_cache)
    d2, dgamma, dbeta = batchnorm_backward(d1, bn_cache)
    d3, dw, db = affine_backward(d2, fc_cache)
    return d3, dw, db, dgamma, dbeta