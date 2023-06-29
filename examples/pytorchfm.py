import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import itertools

import copy




class TorchFM(nn.Module):
    def __init__(self, n=None, k=None, act = 'identity'):
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients

        #self.V = nn.Parameter(torch.rand(n, k, requires_grad=True))
        self.V = nn.Parameter(torch.FloatTensor(n, k).uniform_(-0.1, .1), requires_grad=True)

        self.a= nn.Parameter(torch.FloatTensor(1).uniform_(.1, 3), requires_grad=True)
        self.b = nn.Parameter(torch.FloatTensor(1).uniform_(0.1, 2.5), requires_grad=True)
        self.c= nn.Parameter(torch.FloatTensor(1).uniform_(0.1, 2.5), requires_grad=True)

        self.lin = nn.Linear(n, 1)

        self.n  = n
        self.k = k

        
    def forward(self, x):

        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True) #S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True) # S_2
        
        out_inter = 0.5*(out_1 - out_2)

        #out_inter = 1.35 * torch.nn.functional.tanh( out_inter)  #-2
        #out_inter = 2 * torch.nn.functional.tanh(0.55 * out_inter)  #-2

        out_lin = self.lin(x)

        #out_lin = 1.1 * torch.nn.functional.tanh(out_lin)

        out = out_inter + out_lin 

        #out = 2 * torch.nn.functional.logsigmoid(out) + 1.5
        #out = 2.5 * torch.nn.functional.logsigmoid(out- 1) + 1.3
        #out = self.a.item() * torch.nn.functional.logsigmoid(out- self.b.item()) + self.c.item()
        #out = 1.2 * torch.nn.functional.logsigmoid(out- 1.8) + 1.8

        #best
        #out = 0.6426 * torch.nn.functional.logsigmoid(out- 0.8374) + 0.4480
        #out = 0.46 * torch.nn.functional.logsigmoid(out- 1) + 0.44


        return out

    def return_params(self):
        V = self.V.detach().cpu().numpy().copy()

        return self.lin, V
    






from pyqubo import Binary
import neal

from dimod import ExactSolver
 

class FMTraining:
    def __init__(self, n, k, verbose = False, act = "identity"):
        # Define the model
        self.model = TorchFM(n,k, act)
        self.n = n
        self.k = k
        self.verbose = verbose


    def train(self, X, y, split_size = 0.7, batch_size = 1, n_epochs = 1, minmaxscale=True, standardscale = False, opt = 'LBFGS', LR = 0.01, l2_lambda = 0.001 , l1_lambda = 0):

        # train-test split for model evaluation
        self.X_train_raw, self.X_test_raw, self.y_train_raw, self.y_test_raw = train_test_split(X, y, train_size=split_size, shuffle=True)

        self.standardscale = standardscale
        self.minmaxscale = minmaxscale

        X_train = self.X_train_raw
        X_test = self.X_test_raw

        if self.standardscale or self.minmaxscale:
            # Standardizing data
            if self.standardscale:
                self.scaler = StandardScaler()
            else: 
                self.scaler = MinMaxScaler((-1,1))
            self.scaler.fit(self.y_train_raw)
            self.y_train = self.scaler.transform(self.y_train_raw)
            self.y_test = self.scaler.transform(self.y_test_raw)

        else:
            self.y_train = self.y_train_raw
            self.y_test = self.y_test_raw

        # Convert to 2D PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(self.y_train, dtype=torch.float32).reshape(-1, 1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(self.y_test, dtype=torch.float32).reshape(-1, 1)



        # loss function and optimizer
        loss_fn = nn.MSELoss()  # mean square error
        #optimizer = optim.NAdam(model.parameters(), lr = .03)

        if opt == 'LBFGS':
            optimizer = optim.LBFGS(self.model.parameters(), lr = LR)
        elif opt == 'ADAM':
            optimizer = optim.Adam(self.model.parameters(), lr = LR)
        elif opt == 'NADAM':
            optimizer = optim.NAdam(self.model.parameters(), lr = LR)
        elif opt == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr = LR)
        else: 
            optimizer = optim.Adam(self.model.parameters(), lr = LR)

        


        batch_start = torch.arange(0, len(X_train), batch_size)

        # Hold the best model
        self.best_mse = np.inf   # init to infinity
        self.best_weights = None
        history = []
        history_train = []


        for epoch in range(n_epochs):
            running_loss = 0
            self.model.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                if self.verbose: bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    X_batch = torch.autograd.Variable(X_train[start:start+batch_size], requires_grad =True)
                    y_batch = torch.autograd.Variable(y_train[start:start+batch_size], requires_grad =True)

                    #loss closure function
                    def closure():
                        # Zero gradients
                        optimizer.zero_grad()
                        # Forward pass
                        y_pred = self.model(X_batch)
                        # Compute loss
                        loss = loss_fn(y_pred, y_batch)

                        #l2 regularization
                        l2_norm = sum(p.pow(2.0).sum() 
                            for p in self.model.parameters())

                        l1_norm = sum(torch.abs(p.pow(1.0)).sum() for p in self.model.parameters())
 
                        loss = loss + (l2_lambda * l2_norm) + (l1_lambda * l1_norm)

                        # Backward pass
                        loss.backward()
                        return loss
                    # Update weights
                    optimizer.step(closure)
                    # Update the running loss
                    loss = closure()
                    running_loss += loss.item()
                    # print progress
                    if self.verbose:
                        bar.set_postfix(mse=float(loss))
            # evaluate accuracy at end of each epoch
            self.model.eval()
            y_pred_train = self.model(X_train)
            mse_train = loss_fn(y_pred_train, y_train)
            mse_train = float(mse_train)
            y_pred = self.model(X_test)
            mse = loss_fn(y_pred, y_test)
            mse = float(mse)
            if self.verbose: print('epoch-',epoch,'   mse------',mse, '   mse_train------', mse_train)
            history.append(np.log(mse))
            history_train.append(np.log(mse_train))
            if mse < self.best_mse:
                self.best_mse = mse
                self.best_weights = copy.deepcopy(self.model.state_dict())

        # restore model and return best accuracy
        self.model.load_state_dict(self.best_weights)

        if self.verbose:
            print("MSE: %.2f" % self.best_mse)
            print("RMSE: %.2f" % np.sqrt(self.best_mse))
            plt.plot(history, color = 'green')
            plt.plot(history_train, color = 'red')
            plt.legend(['test', 'train'])
            plt.show()


    def eval(self, num_eval = 20):
        self.model.eval()
        with torch.no_grad():
            # Test out inference with 5 samples from the original test set
            for i in range(num_eval):
                X_sample = self.X_test_raw[i: i+1]

                if self.standardscale or self.minmaxscale:
                    y_pred = self.scaler.inverse_transform(y_pred)

                X_sample = torch.tensor(X_sample, dtype=torch.float32)
                y_pred = self.model(X_sample)
                if self.standardscale or self.minmaxscale:
                    y_pred = self.scaler.inverse_transform(y_pred)

                print(f"{self.X_test_raw[i]} -> {y_pred[0]} (expected {self.y_test[i]})")



    def get_weights(self):
        model_params = copy.deepcopy(self.best_weights)
        return_list =  [np.array(model_params['lin.bias']), np.array(model_params['lin.weight']), np.array(model_params['V'])]
        return return_list

    def return_weights(self):
        params = self.get_weights()
        V = np.array( np.array(params[2]).tolist())
        W = np.array(list(list(np.array(params[1]))[0]), dtype = np.float64)
        bias = np.array(params[0], dtype = np.float64)[0]

        return V, W, bias


    #following 2 methods
    #https://github.com/tsudalab/fmbqm/blob/master/fmbqm/factorization_machine.py

    def triu_mask(self, input_size, F=np.ndarray):
        #Generate a square matrix with its upper trianguler elements being 1 and others 0.

        mask = np.expand_dims(np.arange(input_size), axis=0)
        return (np.transpose(mask) < mask) * 1.0

    def VtoQ(self, V, F=np.ndarray):
        """Calculate interaction strength by inner product of feature vectors.
        """
        print(V.shape)
        #input_size = V.shape[1]
        input_size = V.shape[0]


        #Q = np.dot(np.transpose(V), V) # (d,d)
        Q = np.dot(V , np.transpose(V)) # (d,d)
        #Q =   V @ V.T
        triu = self.triu_mask(input_size)
        #print(triu)
        #return Q * np.triu(np.ones((input_size,input_size)), 1)
        return Q * triu
        


    def get_qubo_hamiltonian_minima(self):
            # define Q_ij = \sum_k V_ik V_jk
        #print('hi')
        params = copy.deepcopy(self.get_weights())

        V = np.array(params[2], dtype = np.float64)
        """
        if self.verbose: 
            print("v.shape :::   ", V.shape)
            print("v:   " , V, "\ntype v:   ", type(V))
            print("v[0]:   " , V[0], "\ntype v[0]:   ", type(V[0]))
            print("v[0[0]]:   " , V[0][0], "\ntype v[0][0]:   ", type(V[0][0]))
        """
        W = np.array(np.array(params[1])[0], dtype = np.float64)
        """
        if self.verbose:
            print("W ----- ", W , "\n type w[0] : ", type(W[0]))
        """
        bias = np.array(params[0], dtype = np.float64)[0]
        """
        if self.verbose: 
            print("B ----- ", bias, "\n type b : ", type(bias))

        """


        """
        Q = np.einsum("ik,jk->ij", V, V)
        print("Q ----- ", Q, "\n type Q : ", type(Q))
        print("Q[0] ----- ", Q[0], "\n type Q[0] : ", type(Q[0]))
        print("Q[0][0] ----- ", Q[0][0], "\n type Q[0][0] : ", type(Q[0][0]))
        """

        Q = V @ V.T

        """
        if self.verbose:
                
            print("Q ----- ", Q, "\n type Q : ", type(Q))
            print("Q[0] ----- ", Q[0], "\n type Q[0] : ", type(Q[0]))
            print("Q[0][0] ----- ", Q[0][0], "\n type Q[0][0] : ", type(Q[0][0]))
        """

        """
        Q = np.zeros((self.n,self.n))

        for i in range(self.n):
            for j in range(self.n):
                Q_ij = 0
                for k in range(self.k):
                    Q_ij += V[i, k] * V[j, k]
                Q[i,j] = Q_ij
        """

        Q = self.VtoQ(V)
        #Q = 1.35 * np.tanh(Q)
        """
        if self.verbose:
            print("Q ----- ", Q, "\n type Q : ", type(Q))
            print("Q[0] ----- ", Q[0], "\n type Q[0] : ", type(Q[0]))
            print("Q[0][0] ----- ", Q[0][0], "\n type Q[0][0] : ", type(Q[0][0]))

        """


        #print("Q: " ,Q)

        binary_len=int(self.n)
        qubo_basis = []
        for i in range(0, binary_len):
            new_string = 'x' + str(i+1)
            qubo_basis.append(Binary(new_string))



        # define qubo Hamiltonian
        H = 0
        for i in range(0, len(Q)):
            for j in range(0, len(Q)):
                H += ( Q[i][j] * qubo_basis[i] * qubo_basis[j])

        for i in range(0, len(W)):
            H+=(W[i]* qubo_basis[i])

        #H+= float(bias) 
 
        #if self.verbose: print(H)
        model = H.compile()
        #if self.verbose: print(model)
        bqm = model.to_bqm()
        #if self.verbose: print(bqm)


        # solve the model
        
        """
        sa= neal.SimulatedAnnealingSampler()
        sampleset = sa.sample(bqm, seed=1234,
                               beta_range=[0.1, 80.2],
                               num_sweeps=20,
                               num_reads = 40,
                               beta_schedule_type='geometric')
        print(sampleset)
        decoded_samples = model.decode_sampleset(sampleset)
        best_sample = min(decoded_samples, key=lambda x: x.energy)
        """
        
        
        
 

        """
        sampleset = ExactSolver().sample(bqm)
        decoded_samples = model.decode_sampleset(sampleset)
        best_sample = min(decoded_samples, key=lambda s: s.energy)
        print(best_sample.energy)
        """
        
        
        sa = neal.SimulatedAnnealingSampler()

        sampleset = sa.sample(bqm, num_sweeps =100, num_reads=1)
        
        #if self.verbose: print(sampleset)
        decoded_samples = model.decode_sampleset(sampleset)
        best_sample = min(decoded_samples, key=lambda x: x.energy)
        
    
        



        print("sample:    ", best_sample.sample)
        print("energy:   ",best_sample.energy)

        return best_sample



    
    def get_qubo_hamiltonian_minimas(self,  num_groups_of_5 = 1):
            # define Q_ij = \sum_k V_ik V_jk
        #print('hi')
        params = copy.deepcopy(self.get_weights())

        V = np.array(params[2], dtype = np.float64)
        """
        if self.verbose: 
            print("v.shape :::   ", V.shape)
            print("v:   " , V, "\ntype v:   ", type(V))
            print("v[0]:   " , V[0], "\ntype v[0]:   ", type(V[0]))
            print("v[0[0]]:   " , V[0][0], "\ntype v[0][0]:   ", type(V[0][0]))
        """
        W = np.array(np.array(params[1])[0], dtype = np.float64)
        """
        if self.verbose:
            print("W ----- ", W , "\n type w[0] : ", type(W[0]))
        """
        bias = np.array(params[0], dtype = np.float64)[0]
        """
        if self.verbose: 
            print("B ----- ", bias, "\n type b : ", type(bias))
        """


        """
        Q = np.einsum("ik,jk->ij", V, V)
        print("Q ----- ", Q, "\n type Q : ", type(Q))
        print("Q[0] ----- ", Q[0], "\n type Q[0] : ", type(Q[0]))
        print("Q[0][0] ----- ", Q[0][0], "\n type Q[0][0] : ", type(Q[0][0]))
        """

        Q = V @ V.T
        """
        if self.verbose:
                
            print("Q ----- ", Q, "\n type Q : ", type(Q))
            print("Q[0] ----- ", Q[0], "\n type Q[0] : ", type(Q[0]))
            print("Q[0][0] ----- ", Q[0][0], "\n type Q[0][0] : ", type(Q[0][0]))
        """

        """
        Q = np.zeros((self.n,self.n))

        for i in range(self.n):
            for j in range(self.n):
                Q_ij = 0
                for k in range(self.k):
                    Q_ij += V[i, k] * V[j, k]
                Q[i,j] = Q_ij
        """

        Q = self.VtoQ(V)
        """
        if self.verbose:
            print("Q ----- ", Q, "\n type Q : ", type(Q))
            print("Q[0] ----- ", Q[0], "\n type Q[0] : ", type(Q[0]))
            print("Q[0][0] ----- ", Q[0][0], "\n type Q[0][0] : ", type(Q[0][0]))
        """


        #print("Q: " ,Q)

        binary_len=int(self.n)
        qubo_basis = []
        for i in range(0, binary_len):
            new_string = 'x' + str(i+1)
            qubo_basis.append(Binary(new_string))



        # define qubo Hamiltonian
        H = 0
        for i in range(0, len(Q)):
            for j in range(0, len(Q)):
                H += ( Q[i][j] * qubo_basis[i] * qubo_basis[j])

        for i in range(0, len(W)):
            H+=(W[i]* qubo_basis[i])

        #H+= float(bias) 
 
        #if self.verbose: print(H)
        model = H.compile()
        #if self.verbose: print(model)
        bqm = model.to_bqm()
        #if self.verbose: print(bqm)


        # solve the model
        
        """
        sa= neal.SimulatedAnnealingSampler()
        sampleset = sa.sample(bqm, seed=1234,
                               beta_range=[0.1, 80.2],
                               num_sweeps=20,
                               num_reads = 40,
                               beta_schedule_type='geometric')
        print(sampleset)
        decoded_samples = model.decode_sampleset(sampleset)
        best_sample = min(decoded_samples, key=lambda x: x.energy)
        """
        
        
        
 

        """
        sampleset = ExactSolver().sample(bqm)
        decoded_samples = model.decode_sampleset(sampleset)
        best_sample = min(decoded_samples, key=lambda s: s.energy)
        print(best_sample.energy)
        """
        
        
        sa = neal.SimulatedAnnealingSampler()


        samples_list = []

        for i in range(0, num_groups_of_5):
            sampleset = sa.sample(bqm,num_sweeps = 1000, num_reads=10)
            #print(sampleset.record['sample'])
            for l in list(sampleset.record['sample']):
                samples_list.append(list(l))
        
        samples_list = [list(tupl) for tupl in {tuple(item) for item in samples_list}]



        decoded_samples = model.decode_sampleset(sampleset)
        best_sample = min(decoded_samples, key=lambda x: x.energy)


        return best_sample, decoded_samples



    def predict(self, X):
        self.model.eval()
        with torch.no_grad():

            X = torch.tensor(X, dtype=torch.float32)
            predictions = self.model(X)

            if self.standardscale or self.minmaxscale:
                predictions = self.scaler.inverse_transform(predictions)

        return predictions
            
            

    def get_predictions(self, X_test, y,  X_test_plot):
        """
        return 2-d array that is [ [X] , [y], [y_pred]] for easy plotting
        """
        x = [[],[],[]]
        self.model.eval()
        with torch.no_grad():
            # Test out inference with 5 samples from the original test set
            for i in range(len(X_test)):
                X_sample = self.X_test_raw[i: i+1]

                if self.standardscale:
                    X_sample = self.scaler.transform(X_sample)

                X_sample = torch.tensor(X_sample, dtype=torch.float32)
                y_pred = self.model(X_sample)

                x[0].append(X_test_plot[i])
                x[1].append(y[i])
                x[2].append(float(y_pred[0][0]))

        return x









#example/testing


def f(x):
    #return x**2 +50*x - 2000
    return 40*np.sin(0.2 * x ) + 5*x**0.5
    return 40*np.sin(0.05 * x ) + 5*x**0.5


from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
x = np.linspace(0, 100, 100)

plt.plot(x, f(x), color='red')

plt.show()


import random

def ConvertToList(string):
    list1 = []
    list1[:0] = string
    return list1
 
def generate_binary_string(n):
    # Generate a random number with n bits
    number = random.getrandbits(n)
    # Convert the number to binary
    binary_string = format(number, '0b')

    binary_String=  ConvertToList(binary_string)

    while(True):
        if(len(binary_String)<n):
            binary_String.insert(0,0)
        else:
            break


    binary_String =[  int(x) for x in binary_String ]

    return binary_String


def convert_to_int(bin):
    bin = np.flip(bin)
    x = 0
    for i in range(0, len(bin)):
        x += bin[i]* i**2
    return x





def test_fm():
    
    # Test the function
    n = 8
    x = generate_binary_string(n)
    print("Random binary string of length {}: {}".format(n, x))
    print("Val = ", str(convert_to_int(x)))


    def generate_training_data(num_bits, num_data):
        train_x = []
        train_y = []
        actuals = []

        for i in range(0, num_data):
            train_x.append(generate_binary_string(num_bits))
            actuals.append(convert_to_int(train_x[i]))
            train_y.append(f(convert_to_int(train_x[i])))


        return train_x, train_y, actuals

    train_x,train_y, actuals = generate_training_data(8, 256)

    print(train_x)
    print(train_y)

    N = len(train_x[0])
    K = 5
    FMtorch= FMTraining(N, K, verbose=True)

    #FMtorch.train(x_train, y_train, split_size = 0.5, batch_size = 10000, n_epochs = 40, standardscale=False, LR = 0.1)
    #FMtorch.train(train_x, np.array(train_y).reshape(-1,1), split_size = 0.8, batch_size = 10000, n_epochs = 200, standardscale=True, LR = 0.02, opt ="LBFGS", l2_lambda=0.002)
    FMtorch.train(train_x, np.array(train_y).reshape(-1,1), split_size = 0.8, batch_size = 10000, n_epochs = 10000, minmaxscale=True, LR = 0.25, opt ="SGD", l2_lambda=0.001)


    predictions = FMtorch.predict(train_x)

        
    plt.plot(actuals, predictions, 'ro')
    plt.plot(actuals, train_y, 'bo')



    def convert_qubo_result_to_plottable_feature(best_sample):

        new_dict = {}
        binary_list = []

        for key in best_sample:
            temp_key = int(key[1:])
            temp_value = best_sample[key]

            new_dict[temp_key] = temp_value

        best_sample = new_dict

        print(best_sample)

        myKeys = list(best_sample.keys())
        myKeys.sort()
        sorted_dict = {i: best_sample[i] for i in myKeys}

        print(sorted_dict)


        for key in sorted_dict:
            binary_list.append(sorted_dict[key])

        _x = convert_to_int(binary_list)
        _y = f(_x)
        


        return _y, _x


    best_sample_new = FMtorch.get_qubo_hamiltonian_minima()

    #get best sample
    best_sample_data = convert_qubo_result_to_plottable_feature(best_sample_new.sample)

    print(best_sample_data)

    #green dot shows what annealing predicted to be the lowest
    plt.plot(best_sample_data[1], best_sample_data[0], 'go')
    plt.show()

    print(FMtorch.model.a)
    print(FMtorch.model.b)
    print(FMtorch.model.c)





#test_fm()