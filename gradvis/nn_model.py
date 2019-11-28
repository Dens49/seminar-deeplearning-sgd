class Base_NNModel():
    """
    Provides an interface to the NN model.
    """

    def __init__(self):
        pass

    def get_parameters(self, filename=None):
        """
        Get the model parameters.
        If filename is provided, it will return the model parameters of the checkpoint. The model is not changed.
        If no arguments are given, it will return the current model parameters.
        Returns a dictionary, comprising of parameter identifiers as keys and numpy arrays as data containers.
        Weights and biases are supposed to have "weight" or "bias" appear in their ID string!

        Args:
            filename: string of the checkpoint, of which the parameters should be returned.
        Return:
            python dictionary of string parameter IDs mapped to numpy arrays.
        """
        raise NotImplementedError("Override this function!")

    def set_parameters(self, parameter_dict):
        """
        Set the model parameters.
        The input dictionary must fit the model parameters!

        Args:
            parameter_dict: python dictionary, mapping parameter id strings to numpy array parameter values.
        """
        raise NotImplementedError("Override this function!")

    def calc_loss(self):
        """
        Calculates the loss of the NN.

        Return:
            The loss, based on the parameters loaded into the model.
        """
        raise NotImplementedError("Override this function!")
