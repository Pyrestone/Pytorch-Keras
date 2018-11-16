import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.autonotebook import tqdm

# Keras-like library for pytorch feedforward NNs because I like Keras
# Author: Marc Uecker (github.com/Pyrestone)


class KerasModel(nn.Module):
    """
    This is an abstract Class containing all the necessary methods to use a PyTorch Module as if it were a Keras model instance.
    
    """

    def __init__(self):
        super(KerasModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Overwrite this method!")
    
    def compile(self,loss,optimizer,lr_scheduler=None):
        self.lr_scheduler=lr_scheduler
        self.optimizer=optimizer
        self.criterion=loss

    def fit(self,x,y,validation_split=0.0,epochs=1,shuffle=True,batch_size=32):
        inputs_tensor=torch.Tensor(x)
        targets_tensor=torch.Tensor(y)
        num_epochs=epochs
        
        train_split=len(x)
        val_split=int((1-validation_split)*train_split)
        
        optimizer=self.optimizer
        criterion=self.criterion
        
        for ep in range(num_epochs):
            if(shuffle):
                p=np.random.permutation(val_split)
            else:
                p=np.arange(val_split)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            loss_sum=0
            loss_num=0
            self.train() # set training mode
            with tqdm(range(0,val_split,batch_size),desc=f"Epoch {ep+1:02d}/{num_epochs:02d}",ncols=200,dynamic_ncols=True) as t:
                for batch_start in t:
                    input=inputs_tensor[p[batch_start:batch_start+batch_size]]
                    target=targets_tensor[p[batch_start:batch_start+batch_size]]
                    # in your training loop:
                    optimizer.zero_grad()   # zero the gradient buffers
                    output = self(input)
                    loss = criterion(output, target)
                    loss_val=loss.data.item()*len(target)
                    loss_sum+=loss_val
                    loss_num+=len(target)
                    t.set_postfix(loss=f"{loss_sum/loss_num:.6f}")
                    #print(f"Loss:\t{loss.data.item()}")
                    loss.backward()
                    optimizer.step()    # Does the update
            if validation_split>0:
                val_loss=self.evaluate(x[val_split:],y[val_split:])
                print(f"Val loss: {val_loss:.6f}")
        self.train()  
    
    def evaluate(self,x,y):
        self.eval()
        inputs_tensor=torch.Tensor(x)
        targets_tensor=torch.Tensor(y)
        output=self(inputs_tensor)
        loss=self.criterion(output,targets_tensor)
        self.train()
        return loss.data.item()
    
    def save(self,path):
        torch.save(self,path)

    def predict(self,x):
        return self(x)
    
class DenseModel(KerasModel):

    def __init__(self,num_inputs,num_outputs,units=64,num_layers=3):
        super(DenseModel, self).__init__()
        self.num_layers=num_layers

        # define layers
        if self.num_layers>1:
            self.fc_in = nn.Linear(num_inputs, units)
        for i in range(num_layers-2):
            #setattr here is necessary because Pytorch doesn't recursively search for layers.
            setattr(self,f"layer_{i}",nn.Linear(units,units))
        if self.num_layers>1:
            self.fc_out = nn.Linear(units, num_outputs)
        else:
            self.fc_out= nn.Linear(num_inputs,num_outputs)

    def forward(self, x):
        if self.num_layers>1:
            x = F.relu(self.fc_in(x))
        for i in range(self.num_layers-2):
            l= getattr(self,f"layer_{i}")
            x= F.relu(l(x))
        x = self.fc_out(x)
        return x

def load_model(path):
    return torch.load(path)




        
        
        
        
        
        
        