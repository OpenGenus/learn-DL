from copy import copy
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
from autograd.variable import Variable
import autograd.variable as av
import autograd as ad
from autograd.blocks.expo import log
log = log()




class Layer(ABC):
    
    def __init__(self,
                 activation,
                 input_shape,
                 name='',
                 seed=None,
                 initialization_method='random'
                 ) -> None:
        # Class variable to be used to track number of layers
        try:
            Layer.nb_layer = Layer.nb_layer +1
        except:
            Layer.nb_layer = 1
        self.init_weight = initialization_method
        self.activation_name = activation
        
        # variable to know the order of the layer in the model
        # and to name the layer in case the user didn't
        self.order = Layer.nb_layer
        
        # input shape is necessary to define the shape of the weights
        self.input_shape = input_shape
        
        
        # The seed is used to ensure reproducibility of the results
        self.seed = seed
        
        
        self.name = (name or 'L')+f'{Layer.nb_layer}'
        
        # This dictionary is used to cache the gradients
        # in the forward pass
        # and used in the optimization for weight update
        self.gradients = {f'dZ{Layer.nb_layer}': [],
                          f'dX{Layer.nb_layer}': [],
                          f'dW{Layer.nb_layer}': [],
                          f'db{Layer.nb_layer}': [],
                          f"g'(Z_{Layer.nb_layer})":[]}
    @abstractmethod
    def forward(self,X,train=True):
        """Method that transfers inputs from current layer to next layer

        Args:
            X (np.ndarray): Features of the dataset
            train (bool, optional): when true, the gradients \
            will be computed in the forward pass. When not,\
            only the inference happens.
        """
        pass
    
    @abstractmethod
    def activation(self,data):
        """Method that introduces non-linearity to the inputs

        Args:
            data (np.ndarray): Array of features
        """
        pass
    
    def update_layer(self,dW,db,alpha=1e-3):
        """Updates the weights of the layer

        Args:
            dW (np.ndarray): Gradients of the loss, with respect to the weights of the layer
            db (np.ndarray): Gradients of the loss, with respect to the bias of the layer
            alpha (float, optional): The Learning rate. Defaults to 1e-3.
        """
        
        self.W = self.W - alpha*dW
        self.bias = self.bias - alpha*db
        
    def clear_gradients(self):
        """Deletes the dictionary of gradients.
        """
        del self.gradients
        self.gradients = {f'dZ{self.order}': [],
                          f'dX{self.order}': [],
                          f'dW{self.order}': [],
                          f'db{self.order}': [],
                          f"g'(Z_{self.order})":[]}
    
    @abstractmethod      
    def backward(self,gradients,alpha=1e-3):
        """

        Args:
            gradients (np.ndarray): gradients from the previous layer
            alpha (float, optional): learning rate. Defaults to 1e-3.
        """
        pass
    
    @abstractmethod    
    def __init_weights__(self,shape):
        """Initalizes the weights

        Args:
            shape (Tuple): Tuple containing the shape of the weights
        """
        
        pass
    
    @abstractmethod
    def __init_bias__(self):
        """Initializes the bias vector
        """
        pass


#----------------------------------------------------------------------------------------#


class FCLayer(Layer):
    
    def __init__(self, nb_neurone,
                 activation,
                 input_shape,
                 name='',
                 init_weight='he',
                 seed=None
                 ) -> None:
        
        super().__init__(activation,
                         input_shape,
                         name,
                         seed)
        self.neurones = nb_neurone
        self.init_weight = init_weight
        self.__init_weights__(shape=(input_shape[1],nb_neurone))
        self.__init_bias__()
        
    
    def forward(self,X,train=True):
        # X = (m,n), W=(n,nb_neurones) => weighted sum = (m,nb_neurones)
        # b= (nb_neurones,)
        self.X = copy(X)
        W = copy(self.W)
        b = copy(self.bias) 
        self.A = np.empty(shape=(X.shape[0],W.shape[1]))
        
        
        for neurone, (w,b_) in enumerate(zip(W.transpose(),b[:,0])):
            
            for i, x in enumerate(X):
                
                z = np.dot(x,w.T) + b_
                activ = self.activation(z)
                self.A[i,neurone] = activ.data 
                
                if train:
                    activ.compute_gradients()
                
                    if not isinstance(self.gradients[f"g'(Z_{self.order})"],np.ndarray):
                        # the gradient matrix is not set yet
                        self.gradients[f"g'(Z_{self.order})"] = np.empty(shape=(X.shape[0],W.shape[1]))
                    
                    self.gradients[f"g'(Z_{self.order})"][i,neurone] = activ.gradient
                    
        return self.A 
    
    def backward(self, gradients, alpha=0.001):
        if (not isinstance(gradients,np.ndarray)):
            raise TypeError(f'The gradient dZ_prev is expected to be a numpy, but got {type(gradients)}')
        
        # compute the gradients according to the equations
        self.gradients[f'dA{self.order}'] = gradients
        
        self.gradients[f'dZ{self.order}'] = np.multiply(gradients,
                                                        self.gradients[f"g'(Z_{self.order})"].T)
    
        
        self.gradients[f'dW{self.order}'] = 1/self.A.shape[0]*np.dot(self.gradients[f'dZ{self.order}'],
                                                                         self.X).T
        # print(f"Layer grads W shape : {self.gradients[f'dW{self.order}']}")
        
        self.gradients[f'db{self.order}'] = 1/self.A.shape[0]*np.sum(self.gradients[f'dZ{self.order}'],axis=1)
       
        # Update the gradients
        self.update_layer(self.gradients[f'dW{self.order}'],
                          self.gradients[f'db{self.order}'],
                          alpha)
        
        dA_minus = np.dot(self.W,self.gradients[f'dZ{self.order}'])
        
        # Clear the gradients
        # Make sure to comment the line below 
        # in case you want to debug the computed gradients
        self.clear_gradients()
        return dA_minus    
    def activation(self, data):
    
        if self.activation_name == 'relu':
            return self.__leaky_relu__(data)
        
        elif self.activation_name == 'sigmoid':
            return self.__sigmoid__(data)
        
        
    def __leaky_relu__(self,data):
        
        zero = Variable(0.0)
        x = Variable(data)
        activ = x if x> zero else 0.1*x
        
        return activ
    
    def __sigmoid__(self, data):
        
        z = Variable(data)
        activation = 1/(1 + ad.exp(-1*z))
        return activation
    
    def __random_initialization__(self,shape):
       
        # self.W = self.seed.normal(size=shape)
        self.W = self.seed.normal(size=shape)
    
    def __xavier_initialization__(self,shape):
        
        n = shape[0]
        lower, upper = -(1.0 / np.sqrt(n)), (1.0 / np.sqrt(n))
        self.W = self.seed.uniform(low=lower,high=upper,size=shape) #+np.finfo(float).eps
        
    def __init_bias__(self):
        self.bias = self.seed.uniform(np.finfo(float).eps, 1e-1,size=(self.neurones,1))
        # self.bias = np.zeros(shape=(self.neurones,1))
    
    def __init_weights__(self,shape):
        
        if self.init_weight == "xavier":
            self.__xavier_initialization__(shape)
        else:
            self.__random_initialization__(shape)
        
        # print(f'W.shape={self.W}')
        
        
        
#----------------------------------------------------------------------------#

class Model:
    
    def __init__(self,input_shape:List,
                      neurones: List[int],
                      activations: List[str],
                      init_weights: Optional[List[str]]=None) -> None:
        self.seed =  np.random.default_rng(900)
        self.layers = self.__create_layers__(input_shape,
                                             neurones,
                                             activations,
                                             init_weights)
    
    def __create_layers__(self,input_shape,
                               neurones,
                               activations,
                               init_weights=None
                              ):
        
        layers = []
        
        init_weights = init_weights or ['he']*len(activations)
        # init_weights[-1] = 'xavier'
        
        for nb_neurone,activation in zip(neurones,activations):
            
            if activation == 'sigmoid':
                l = FCLayer(nb_neurone,activation,input_shape,init_weight='xavier',seed=self.seed)
            else:    
                l = FCLayer(nb_neurone,activation,input_shape,seed=self.seed)
            # print(f'\n\nLayer: {l.name}')
            layers.append(l)
            input_shape = (input_shape[0],nb_neurone)
        return layers
    
    def fit(self,X,Y,
            batch_size,
            nb_iterations=1000,
            learning_rate=1e-4,
            val_rate=0.10
            ):
        X_,Y_ = self.__batch_data__(X= X,
                                    Y= Y,
                                    batch_size=batch_size)
        
        (X_train,Y_train),(X_val,Y_val) = self.__split_data__(x_batches= X_,
                                              y_batches= Y_,
                                              rate=1-val_rate)
        
        losses = []
        accuracies = []
        for iter in range(nb_iterations):
            tmp_losses = []
            tmp_acc = []
            for x_batch,y_batch in zip(X_train,Y_train):
                
                y_hat = self.forward_pass(x_batch)
               
                loss, grads = self.loss(np.squeeze(y_hat),np.squeeze(y_batch))
                
                tmp_losses.append(loss)
                
                self.backword_pass(grads,alpha=learning_rate)
                
                # Accuracy
                y_class = (y_hat >= 0.5)*1
                acc = self.accuracy(y_class,y_batch )
                tmp_acc.append(acc)
                
            train_loss = np.array(tmp_losses).mean()
            train_accuracy = np.array(tmp_acc).mean()
            
            
            vtmp_losses = []
            vtmp_acc = []
            for x_batch,y_batch in zip(X_val,Y_val):
                y_hat = self.predict_proba(x_batch)
                
                loss = self.loss(np.squeeze(y_hat),np.squeeze(y_batch),train=False)
                
                vtmp_losses.append(loss)
                
                # accuracy
                y_class = (y_hat >= 0.5)*1
                acc = self.accuracy(y_class,y_batch )
                vtmp_acc.append(acc)
            val_loss = np.array(vtmp_losses).mean()
            val_accuracy = np.array(vtmp_acc).mean()
            
            print(f'Iteration: {iter}, train loss: {train_loss}, train accuracy : {train_accuracy},\
                val loss: {val_loss}, val accuracy: {val_accuracy}')
            
            losses.append((train_loss,val_loss))
            accuracies.append((train_accuracy,val_accuracy))
        return losses, accuracies
        
    
    def forward_pass(self,X,train=True):
        out = copy(X)
        for l in self.layers:
            out = l.forward(out,train)
            # print(f'\n\nA_{l.name}: {out.shape}')
            
        return out
    
    def backword_pass(self, grads,alpha):
        g = copy(grads)
        
        for layer in reversed(self.layers):
            g = layer.backward(g,alpha=alpha)  
    
    def loss(self,y_hat, y, train=True):
        # implements log loss (Binary cross entropy)
        nb_samples =y_hat.shape[0]
        y_pred, y_true = av.Variable.multi_variables(y_hat, y)
        probabilities = ad.multiply(log(y_pred+ np.finfo(np.float32).eps),y_true) + ad.multiply(log(1-y_pred + np.finfo(np.float32).eps),(1-y_true))
        
       
        results = ad.sum_elts(-probabilities)/nb_samples
        # print(f'results.data: {results.data}')
        if train:
            results.compute_gradients()
            return results.data, results.gradient[0]
        
        return results.data
    
    def accuracy(self,y_pred, y_true):
        
        # accuracy_score = #(true predictions)/total_predictions
        
        return (y_pred == y_true).sum()/y_pred.shape[0]    
    
    def predict_proba(self,data):
        
        # Predicts the probability of each entry
        
        return self.forward_pass(data,train=False)
    
    def predict(self,data):
        
        # Predicts the class of each entry
        
        return (self.predict_proba(data) >= 0.5)*1
    
    def __batch_data__(self,X,Y,batch_size):
        
        # Divides a dataset to batches of batch_size
        
        x_batches = [X[i: i+batch_size,:] for i in range(0,X.shape[0],batch_size)]
        y_batches = [Y[i: i+batch_size,:] for i in range(0,Y.shape[0],batch_size)]
        return x_batches, y_batches
    
    def __split_data__(self,x_batches,y_batches,rate,shuffle=True):
         
        # Splits the data into traning and validation set
        if not isinstance(rate,int) and not isinstance(rate,float):
            raise TypeError(f'rate should be an integer between 0 and 100\
                or a float between 0 and 1. Got rate: {rate} of type : {type(rate)}')
            
        if isinstance(rate,float):
            rate = rate *100
        
        x , y = np.array(x_batches), np.array(y_batches)
        indx = int(rate*len(x_batches)//100)
        
        
        if  shuffle:
            perm = self.seed.permutation(len(x_batches))
            x, y = x[perm], y[perm]
        
        
        x_train, y_train = x[:indx], y[:indx]
        x_val, y_val = x[indx:], y[indx:]
        # print(f'x_train.shape: {x_train.shape}, x_val: {x_val.shape}, ytrain: [{y_train.shape}], y_val: {y_val.shape}')
        
        return (x_train, y_train), (x_val,y_val)
    