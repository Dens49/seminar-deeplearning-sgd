import torch
from .nn_model import Base_NNModel


class PyTorch_NNModel(Base_NNModel):
    """
    Provides an interface to the PyTorch NN model.
    """

    def __init__(self, model, trigger_fn, filename):
        super(PyTorch_NNModel, self).__init__()
        #self.parameter = self._torch_params_to_numpy(torch.load(filename))
        self.model = model
        self.parameter = self.get_parameters(filename)
        self.trigger_fn = trigger_fn
        self.set_parameters(self.parameter)

    def get_parameters(self, filename=None):
        if filename is None:
            return self.parameter
        else:
            return self._torch_params_to_numpy(torch.load(filename))
            # self.model.load_state_dict(torch.load(filename))
            # print(self.model)
            # return self._torch_params_to_numpy(dict(self.model.named_parameters()))

    def get_param_vec(self):
        return np.concatenate([ar.flatten() for ar in list(parameter.values())], axis=None)

    def set_parameters(self, parameter_dict):
        # if self.parameter.dims != parameter.dims or any(self.parameter.shape != parameter.shape):
        #    raise RuntimeError("New parameter shape is not the same as old one!")
        self.model.load_state_dict(self._numpy_params_to_torch(parameter_dict))

    def calc_loss(self):
        # update Dennis Bystrow: use the model we already know
        # instead of relying on a global variable
        return self.trigger_fn(self.model)

    def _numpy_params_to_torch(self, parameter):
        new_param = dict()
        for key, val in parameter.items():
            new_param[key] = torch.tensor(val)
        return new_param

    def _torch_params_to_numpy(self, parameter):
        new_param = dict()
        for key, val in parameter.items():
            new_param[key] = val.cpu().detach().numpy()
        return new_param
