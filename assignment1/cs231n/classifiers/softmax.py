import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train=X.shape[0]
  num_class=W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  #####
  #code
  #####
  
  for i in range(num_train):
    # cal_score=X[i].dot(W)
    # print(X.shape)
    # print(W.shape)
    # print(cal_score.shape)
    # cal_score_norm=np.divide(cal_score, np.sum(cal_score,axis=0))
    # print(cal_score_norm.shape)
    # predicted=cal_score_norm[i,y[i]]
    # predicted_log=-np.log(predicted)
    # loss+=predicted_log
  
    score=X[i].dot(W)
    score-=np.max(score)
    sum_row=np.sum(np.exp(score))
    norm = lambda k: np.exp(score[k])/sum_row
    loss +=-np.log(norm(y[i]))

    for j in range(num_class):
       norm_j=norm(j)
       dW[:,j]+=(norm_j - (j==y[i]))*X[i]

  dW /=num_train
  dW+=reg*np.sum(W*W)
  loss/=num_train
  loss+=0.5*reg*np.sum(W*W)

  #####
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train=X.shape[0]
  num_class=W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  ##########
  #CODE
  #########
  score=X.dot(W)
  score=score-np.max(score,axis=1,keepdims=True)
  e_score=np.exp(score)
  e_score_sum=np.sum(e_score,axis=1)
  e_score_sum=e_score_sum.reshape(len(e_score_sum),1)
  e_score_norm=np.divide(e_score,e_score_sum)
  predicted=e_score_norm[np.arange(num_train),y]
  predicted_log=-np.log(predicted)
  loss=np.sum(predicted_log,axis=0)
  loss/=num_train
  loss+=0.5*reg*np.sum(W*W)

  mat_correct_scores=np.zeros_like(e_score_norm)
  mat_correct_scores[np.arange(num_train),y]=1
  dloss_by_dscore=e_score_norm- mat_correct_scores
  dW=X.T.dot(dloss_by_dscore)
  dW/=num_train
  dW+=reg*W
  #########
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

