import numpy as np
from time import time
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    # Assuming a shape identical to the input image for the conv layer output
    self.params['W2'] = weight_scale * np.random.randn(num_filters * H * W // 4, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    t0 = time()    
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    t1 = time()
    # print(t1-t0)
    N,C,H,W = X.shape
    z1 , conv_fwd_cache = conv_forward_naive(X,W1,b1,conv_param)
    
    t2 = time()
    # print("conv fwd time:")
    # print(t2-t1)
    a1 , z1_cache = relu_forward(z1)
    a1_after_pool , pool_cache = max_pool_forward_naive(a1,pool_param)
    t3 = time()
    # print(t3-t2)    
    z2 , affine1_cache = affine_forward(a1_after_pool,W2,b2)
    a2 , z2_cache = relu_forward(z2)
    t4 = time()
    # print(t4-t3)
    z3 , affine2_cache = affine_forward(a2,W3,b3)
    t5 = time()
    # print(t5-t4)
    scores = z3 

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    
    loss , dout = softmax_loss(z3,y)
    loss += self.reg*0.5*(np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))

    dx3,dw3,db3 = affine_backward(dout ,affine2_cache)
    dx3_relu = relu_backward(dx3 , z2_cache)

    dx2,dw2,db2 = affine_backward(dx3_relu,affine1_cache)

    dx2_pool = max_pool_backward_naive(dx2,pool_cache)

    dx2_relu = relu_backward(dx2_pool , z1_cache)
    
    t10=time()
    dx1,dw1,db1 = conv_backward_naive(dx2_relu ,conv_fwd_cache)
    t11 = time()
    # print("conv_backward_time")
    # print(t11-t10)
    
    grads['W3'], grads['b3'] = dw3 + self.reg*W3, db3
    grads['W2'], grads['b2'] = dw2 + self.reg*W2, db2
    grads['W1'], grads['b1'] = dw1 + self.reg*W1, db1
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
