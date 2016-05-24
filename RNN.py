
# The script MUST contain a function named azureml_main
# which is the entry point for this module.
#
# The entry point function can contain up to two input arguments:
#Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
import numpy as np
import pandas as pd


def azureml_main(dataframe1 = None, dataframe2 = None):

        # data I/O
    
    data = dataframe1#load data
    input_text = data['body']
    print input_text
    charspeach=  '.'.join(line or "" for line in input_text)#.join(input_text)#remove all withe spaces
    #print chars
    #raise NameError('HiThere')
    charspeach=charspeach.decode('utf-8','ignore').encode("utf-8")# get rid of encoding hell
    
    #print chars
    #raise NameError('HiThere')
    
    chars = list(set(charspeach))# get distinct encoded characters from entry text
    #raise NameError('HiThere')
    data_size, vocabulary_size = len(input_text), len(chars)
    print 'data has %d characters, %d unique.' % (data_size, vocabulary_size)
    char_to_index = { ch:i for i,ch in enumerate(chars) }
    index_to_char = { i:ch for i,ch in enumerate(chars) }
    #raise NameError('HiThere')
    
    # hyperparameters
    hidden_size = 100 # size of hidden layer of neurons
    sequence_length = 20 # number of steps to unroll the RNN for
    learning_rate = 5e-2
    
    # build the model
    Wxh = np.random.randn(hidden_size, vocabulary_size)*0.01 # input to hidden
    Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
    Why = np.random.randn(vocabulary_size, hidden_size)*0.01 # hidden to output
    bh = np.zeros((hidden_size, 1)) # hidden bias
    by = np.zeros((vocabulary_size, 1)) # output bias
    
    def lossFun(inputs, targets, hprev):
      """
      inputs,targets are both list of integers.
      hprev is Hx1 array of initial hidden state
      returns the loss, gradients on model parameters, and last hidden state
      """
      xs, hs, ys, ps = {}, {}, {}, {}
      hs[-1] = np.copy(hprev)
      loss = 0
      # forward pass
      for t in xrange(len(inputs)):
        xs[t] = np.zeros((vocabulary_size,1)) # encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
        ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
        loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
      # backward pass: compute gradients going backwards
      dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
      dbh, dby = np.zeros_like(bh), np.zeros_like(by)
      #print 'start dhnext'
      #print hs
      #print 'end dhnext'
      dhnext = np.zeros_like(hs[0])
      for t in reversed(xrange(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)
      for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
      return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
    
    
    def generate(h, seed_index, n):
      """ 
      generate a sequence of integers from the model 
      h is memory state, seed_index is seed letter for first time step
      """
      x = np.zeros((vocabulary_size, 1))
      x[seed_index] = 1
      indexs = []
      for t in xrange(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        index = np.random.choice(range(vocabulary_size), p=p.ravel())
        x = np.zeros((vocabulary_size, 1))
        x[index] = 1
        indexs.append(index)
      return indexs
    
    n, p = 0, 0
    mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
    smooth_loss = -np.log(1.0/vocabulary_size)*sequence_length # loss at iteration 0
    df = pd.DataFrame(columns=('loss','Iterat' ,'Speach'))#creat a new dataframe to store results in it
    counter = 1
    #raise NameError('HiThere')
    while counter<1000000:#change here to iterate less or more
      # prepare inputs (we're sweeping from left to right in steps sequence_length long)
      if p+sequence_length+1 >= len(input_text) or n == 0: 
        hprev = np.zeros((hidden_size,1)) # reset RNN memory
        p = 0 # go from start of data
      inputs = [char_to_index[ch] for ch in charspeach[p:p+sequence_length]]#get index of the entry from postion p to position p+sequence_length
      targets = [char_to_index[ch] for ch in charspeach[p+1:p+sequence_length+1]] # get index of the entry from postition p+1 to p+sequ+1 (our target is to predict the charachter after the sequence )
      
      
      # forward sequence_length characters through the net and fetch gradient
      if(len(inputs)>0):# making sure we have somthing to analyze
          
          if n % 100 == 0 :
            generated_indexes = generate(hprev, inputs[0], 200)
            txt = ''.join(index_to_char[index] for index in generated_indexes)
             
            
          loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
          smooth_loss = smooth_loss * 0.999 + loss * 0.001
          
      if n % 100 == 0: 
          df.loc[counter] =[loss,n,txt.decode('utf-8','ignore').encode("utf-8")]
          counter = counter + 1
      
      # perform parameter update with Adagrad help to optimise gradiant calculation : http://sebastianruder.com/optimizing-gradient-descent/
      for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                    [dWxh, dWhh, dWhy, dbh, dby], 
                                    [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
    
      p += sequence_length # move to the following sequence postion actual is p next is p+sequence_length
      n += 1 # iteration counter check to see if we have to generate a speech  ( every 100l lines)
      
    
    # Return value must be of a sequence of pandas.DataFrame
    return df
    
    
    
    