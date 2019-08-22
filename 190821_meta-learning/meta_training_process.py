import torch


class MetaLearner(nn.Module):
    def __init__(self,model):
        super(MetaLearner,self).__init__()
        self.weights = Parameter(torch.Tensor(1,2))

    def forward(self,forward_model,bachward_model):
        """
        Forward optimizer with a simple linear neural net
        Inputs:
            forward_model: PyTorch module with parameters gradient populated
            backward_model: PyTorch module identical to forward_model (but without gradients)
            updated at the Parameter level to keep track of the computation graph for meta-backward pass
        """
        f_model_iter = get_params(forward_model)
        b_model_iter = get_params(bachward_model)
        for f_param_tuple,b_param_tuple in zip(f_model_iter,b_model_iter):
            (module_f,name_f,param_f) = f_param_tuple
            (module_b, name_b, param_b) = b_param_tuple
            inputs = Variable(torch.stack([param_f.grad.data, param_f.data], dim=-1))
            # Optimization step: compute new model parameters, here we apply a simple linear function
            dW = F.linear(inputs, self.weights).squeeze()
            param_b = param_b + dW
            # Update backward_model (meta-gradients can flow) and forward_model (no need for meta-gradients).
            module_b._parameters[name_b] = param_b
            param_f.data = param_b.data

def train(forward_model,bachward_model,optimizer,meta_optimizer,train_data,meta_epochs):
    """
    train a meta-learner
    :param forward_model and bachward_model:Two identical PyTorch modules (can have shared Tensors)
    :param optimizer:a neural net to be used as optimizer (an instance of the MetaLearner class)
    :param meta_optimizer:an optimizer for the optimizer neural net, e.g. ADAM
    :param train_data:an iterator over an epoch of training data
    :param meta_epochs:meta-training steps
    :return:
    """
    for meta_epoch in range(meta_epochs):
        optimizer.zero_grad()
        losses = []
        for inputs,labels in train_data: # Meta-forward pass
            forward_model.zero_grad() # # Forward pass
            inputs = Variable(inputs)
            labels = Variable(labels)
            output = forward_model(inputs)
            loss = loss_func(output,labels) # Compute loss
            losses.append(loss)
            loss.backward() # # Backward pass to add gradients to the forward_model
            optimizer(forward_model,bachward_model) # optimizer step
        meta_loss = sum(losses) # Compute a simple meta-loss
        meta_loss.backward() # Meta-backward pass
        meta_optimizer.step() # Meta-optimizer step


