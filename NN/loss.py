import numpy as np

class Loss:

    def remember_trainable_layers(self,trainable_layers):
        self.trainable_layers = trainable_layers

    def regularization_loss(self):

        regularization_loss = 0

        for layer in self.trainable_layers:

            if layer.weights_regularizer_l1 > 0:
                regularization_loss += layer.weights_regularizer_l1*np.sum(np.abs(layer.weights))

            if layer.weights_regularizer_l2 > 0:
                regularization_loss += layer.weights_regularizer_l2*np.sum(layer.weights * layer.weights)

            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1*np.sum(np.abs(layer.biases))

            if layer.weights_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2*np.sum(layer.biases* layer.biases)
            
        return regularization_loss
        
    def calculate(self,output,y,*, include_regularization=False):
        sample_losses= self.forward(output,y)

        data_loss = np.mean(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()

class Loss_CategoricalCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-5, 1.0 - 1e-5)
      
        # print(y_true.shape)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped [range(samples), y_true]
        
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood
    
    def backwards(self,d_output,y_true):
        samples =len(d_output)
        labels = len(d_output[0])
        # print(d_output)
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.d_inputs = -y_true/d_output
        self.d_inputs = self.d_inputs /samples
    

