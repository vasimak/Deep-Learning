import numpy as np
from layer import *
from loss import *
from softmax import *
from optimizer  import *

class Object():
    
    def __init__(self):

        self.layers =[]
      

    def add(self,layer):
        self.layers.append(layer)

    def set(self,*,loss, optimizer,accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

  
    def train(self,X,y,*,epochs=1, print_every =1 , validation_data=None):

        self.accuracy.init(y)

        for epoch in range(1, epochs+1):

            output = self.forward(X,training =True)

            data_loss, regularization_loss = self.loss.calculate(output,y,include_regularization=True)
            loss = data_loss + regularization_loss

            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions,y)

            
            self.backwards(output, y)
            self.optimizer.pre_update_parameters()
            for layer in self.trainable_layers:
                self.optimizer.update_parameters(layer)
                self.optimizer.post_update_parameters()

            if not epoch %  print_every:
                print(f'epoch: {epoch}, ' +
                f'loss: {loss:.3f} (' +
                f'data_loss: {data_loss:.3f} (' +
                f'reg loss : {regularization_loss: .3f}), ' +
                f'accuracy: {accuracy:.3f}, ' +
                f'optimizer: {self.optimizer.current_learning_rate}')

        if validation_data is not None:
            X_val, y_val = validation_data

            output = self.forward(X_val,training=False)

            loss = self.loss.calculate(output, y_val)

            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            print(f'validation,' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}')
        

        

    def finalize(self):

        self.input_layer = Layer_Input()

        layer_count = len(self.layers)
        self.trainable_layers=[]

        for i in range(layer_count):

            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            elif i < layer_count -1: 
                
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation =self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layers(
            
            self.trainable_layers
        )
    def forward(self,X,training):
        
        self.input_layer.forward(X,training)

        for layer in self.layers:
            layer.forward(layer.prev.output,training)

        return layer.output

    def backwards(self,output,y):

        self.loss.backwards(output,y)

        for layer in reversed(self.layers) :
            layer.backwards(layer.next.d_inputs)
        

class Accuracy:

    def calculate(self, predictions,y):

        comparisons = self.compare(predictions,y)

        accuracy = np.mean(comparisons)

        return accuracy


class Accuracy_Categorical(Accuracy):
    def init(self,y):
        pass
    
    def compare(self,predictions,y):
        if len(y.shape)== 2:
            y = np.argmax(y,axis=1)
        
        return predictions==y

X,y = spiral_data(100,10)

model = Object()

model.add(Layer_Dense(2, 64,weights_regularizer_l2=5e-4,bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Dropout_Layer(0.1))
model.add(Layer_Dense(64, 10))
model.add(Activation_Softmax())

model.set(
    loss = Loss_CategoricalCrossentropy(),
    optimizer = Optimizer_Adam(learning_rate=0.005,decay=1e-3),
    accuracy = Accuracy_Categorical()

)
model.finalize()

model.train(X,y, epochs=10000,print_every=100)