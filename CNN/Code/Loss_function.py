import numpy as np

class Loss:
    def remember_trainable_layers(self,trainable_layers):
        self.trainable_layers = trainable_layers
        
    def calculate(self,output,y):
        sample_losses= self.forward(output,y)

        data_loss = np.mean(sample_losses)


        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 10e-3, 1.0 - 10e-3)
      
        # print(y_true.shape)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped [range(samples), y_true.T]
        
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood
    
    def backwards(self,d_output,y_true):
        samples =len(d_output)
        labels = len(d_output[0])
        # print(d_output)
        # print(y_true)
        if len(y_true.shape) == 1:
            print('yehaw')
            y_true = np.eye(labels)[y_true]

        self.d_inputs = -y_true.T/d_output
        # print(self.d_inputs)
        self.d_inputs = self.d_inputs /samples
    

