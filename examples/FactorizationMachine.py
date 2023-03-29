import numpy as np

class FactorizationMachine:
    def __init__(self, N, K):
        self.N = N  #<== number of features of the input, e.g. x = [1,0,0] has 3 features
        self.K = K #<== latent space of the interaction matrix

        self.V = np.random.rand(N, K) #<==interaction matrix
        self.w = np.random.rand(N) #<== weights for linear portion
        self.w0 = np.random.rand() #<== bias for linear portion
    

    def fit(self, x_train , y_train ,   alpha = 0.005, num_epochs = 100, lambda_w=0.001, lambda_v = 0.001):

        #l2 regularization to help prevent overfitting
        #l2_regularization = np.sum(np.add(np.multiply(lambda_w, np.power(self.w, 2)) , np.multiply(lambda_v, np.power(self.V, 2))))
        #print(l2_regularization)

        for epoch in range(0, num_epochs):
            gradient_vector = np.zeros_like(self.V) #<== initialize gradient vector
            gradient_W = np.zeros_like(self.w)
            gradient_w0 = 0
            mse = 0 #<== intialize mean squared error

            for i in range(0, x_train.shape[0]):
                y_P = self.predict( x_train[i,:]) #<== current prediction
                sE = (y_train[i] - y_P) ** 2 #<== squared error of given input with current V
                mse += sE  #<== accumulate mean squared error from this particular input

                grad = self.error_gradient( x_train[i,:], y_train[i])



                gradient_vector += grad[0] / self.N #<== accumulate derivative of mse in the gradient
                gradient_W += grad[1] / self.N
                gradient_w0 += grad[2] / self.N
                
            if(epoch%100 == 0):
                print(F'MSE: {mse}')
            self.V = self.V - gradient_vector * alpha
            self.w = self.w - gradient_W * alpha
            self.w0 = self.w0 - gradient_w0 * alpha


        
    def predict(self, x):
        """ Function to predict the output y based on the input x
        
        Arguments
        ---------
        v : NxK numpy array of floats
            the interaction matrix
            
        x : 1xN bitstring
            the input 
            
        Returns
        -------
        y : float
            the output
            
            
        Notes
        -----
        See Equation (5)
        
        """

        y = 0 #<== initialize y to zero
        
        
        #linear term
        linear = x @ self.w
        #print('linear term: ', linear)
        
        #interactions term
           # Loop based evaluation - slow
        #y2 = 0 #<== initialize y to zero
        #for k in range(K):
        #    for i in range(N):
        #        for j in range(N):
        #            y2 += v[i,k] * v[j, k] * x[i] * x[j]
    
    
        # einsum version - faster
        y = np.einsum('ik, jk,i,j->', self.V, self.V, x, x, optimize=True)
                    
                    
        #print("y: ", y)
        
        output = y + linear + self.w0
        
        #print('output: ',output)
            
        return output

    def partial_y_partial_v_ik(self, x, i, k):
        """Function to compute the derivative of the output y with respect to interaction matrix element vik
        
        Arguments
        ---------
        v : NxK numpy array of floats
            the interaction matrix
            
        x : 1xN bitstring
            the input
            
        i : integer
            the feature index of the interaction matrix element 
            
        k : integer
            the latent space index of the interaction matrix element
            
        Returns
        -------
        deriv : float
            the derivative of the output vy with respect to the interaction matrix element vik
        
        Notes
        -----
        See Equation (4)
        """
        # Loop based derivative... slower
        #deriv = 0 #<== initialize derivative to zero
        #for j in range(len(x)):
        #    deriv += 2 * v[j,k] * x[i] * x[j] #<== accumulate derivative term
        #assert np.isclose(deriv, deriv2)
        # using np.dot to compute derivative should be faster!
        
        xi = x[i]
        vjk = np.copy(self.V[:,k])
        deriv = 2 * xi * np.dot(vjk, x)        
        return deriv


    def partial_y_w_i(self, x, i: int):
        """function to compute derivative of the output with respect to linear term element wi
        
        w1: weight vector for linear term
        
        x: 1 xN bitstring th einput
        
        i: integer
            element of this vector
        
            Returns
        -------
        deriv : float
            the derivative of the output y with respect to the interaction matrix element xi
        
        Notes
        -----
        See Equation (4)
        """
        
        deriv = x[i]
        
        return deriv
        
        
        

    def error_gradient(self, x, y_T):
        """Function to compute the gradient matrix of (y_T - y(x))^2 wrt to all elements of v_ik
        
        Arguments
        ---------
        v : NxK numpy array of floats
            the interaction matrix
            
        x : 1xN bitstring
            training input 
            
        y_T : float
            training output
            
        Returns
        -------
        grad_V : NxK numpy array of floats
            the gradient matrix of (y_T - y(x))^2 wrt to all elements of v_ik
            
        Notes
        -----
        See Equation (3)
        
        """
        w_len = len(self.w)
        K = len(self.V[0,:]) #<== dimension of latent space
        N = len(self.V[:,0]) #<== number of features
        y_P = self.predict( x) #<== current prediction from input x and current interaction matrix
        grad_v = np.zeros((N, K)) #<== initialize the gradient matrix
        grad_w = np.zeros((w_len))
        
        deriv_w0 = 1
        
        for i in range(w_len):
            deriv_wi = self.partial_y_w_i( x, i)
            grad_w[i] = -2 * y_T * deriv_wi + 2 * y_P * deriv_wi
        
        for i in range(N):
            for k in range(K):
                deriv_vik = self.partial_y_partial_v_ik(x, i, k) #<== compute derivative of output y wrt v_ik
                grad_v[i, k] = -2 * y_T * deriv_vik + 2 * y_P * deriv_vik #<== accumulate gradient matrix element
                
        grad_w0 = -2 * y_T * deriv_w0 + 2 * y_P * deriv_w0
                
                
        return [grad_v, grad_w, grad_w0]
            
            





import tensorflow as tf
class FactorizationMachineTF(tf.keras.Model):
  #tensorflow factorization machine to see if using some different optimizers such as ADAM will yield better results


  def __init__(self, N: int, K: int, name=None): 
    super().__init__(name=name)

    # number of features
    self.N = tf.Variable(N, trainable = False ,name= 'N')
    # number of latent factors
    self.K = tf.Variable(K, trainable = False ,name= 'K')


    #bias and weights
    self.w0 = tf.Variable(tf.zeros([1]) , trainable = True, name = "bias")
    print(self.w0)

    self.W = tf.Variable(tf.zeros([N]), trainable = True, name = "weights")
    print(self.W)

    #interactions
    self.V = tf.Variable(tf.random.normal([K, N], stddev=0.01), trainable = True, name= "interactions")
    print(self.V)

  def __call__(self, X, training = False):

        #predicted y value
        self.y_hat = tf.Variable(tf.zeros([int(self.N), 1]), trainable=False)

        linear_terms = tf.add(self.w0, tf.math.reduce_sum(tf.multiply(self.W, X), axis = -1, keepdims=True))

        interactions = (tf.multiply(0.5,
                tf.math.reduce_sum(
                    tf.subtract(
                        tf.pow( tf.matmul(X, tf.transpose(self.V)), 2),
                        tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(self.V, 2)))),
                    axis = -1, keepdims=True)))
    

        self.y_hat = tf.add(linear_terms, interactions)


        return self.y_hat


"""

#some quick examples using both factorization machines

fm1 = FactorizationMachineTF(2,100, name="fm")


fm1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-2),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mean_absolute_error'], run_eagerly=True)


X_trn = [[2.0,2.0], [2.0, 3.0] , [3.0 , 3.0] , [3.0, 4.0] , [4.0, 4.0], [4.0, 5.0]]
Y_trn = [ 4, 6, 9, 12, 16, 20]

fm1.fit(X_trn, Y_trn, epochs = 10, batch_size = 6)
print(fm1.predict([[2.0,3.0]]))
print(fm1.predict([[5.0,5.0]]))


fm = FactorizationMachine(2, 100)

fm.fit(np.array(X_trn), np.array(Y_trn), alpha = 0.0000005, num_epochs=1000)


#using either factorization machine will give similar results







class FactorizationMachineKormos:
    def __init__():
        pass

"""