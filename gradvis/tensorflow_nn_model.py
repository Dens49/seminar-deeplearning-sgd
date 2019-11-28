import tensorflow as tf
from .nn_model import Base_NNModel


class Tensorflow_NNModel(Base_NNModel):
    """
    Provides an interface to the Tensorflow NN model.
    """

    def __init__(self, model, trigger_fn, filename, number_of_steps=2):
        print("Build Tensorflow Model...")
        super(Tensorflow_NNModel, self).__init__()
        print("Making Saver...")
        self.saver = tf.train.Saver()  # Saver used to restore parameters from file

        # Create Session
        hooks = []  # optinal hooks
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session_creator = tf.train.ChiefSessionCreator(
            checkpoint_filename_with_path=filename, config=config)
        print("Making Session...")
        self.mon_sess = tf.train.MonitoredSession(
            session_creator=session_creator, hooks=hooks)

        # maximum number of iteration steps per evaluation
        self.number_of_steps = number_of_steps
        self.model = model
        self.total_loss = trigger_fn
        print("Initializing Parameters...")
        self.parameter = self._tf_params_to_numpy()
        # TODO: make dict of numpy arrays from self.parameter
        print("Done.")

    def get_parameters(self, filename=None):
        if filename is None:
            return self.parameter
        else:
            self.saver.restore(self.mon_sess, filename)
            tmp_params = self._tf_params_to_numpy()
            # restore old state, since getter is not changing model
            self.set_parameters(self.parameter)
            self.parameter = tmp_params
            return self.parameter

    def set_parameters(self, parameter_dict):
        for var in tf.trainable_variables():
            var.load(parameter_dict[var.name[:-2]], self.mon_sess)

    def calc_loss(self):
        average_loss = 0
        for i in range(self.number_of_steps):
            current_loss = self.mon_sess.run(self.total_loss)
            # print(np.argmax(label,axis=1))
            average_loss += current_loss
        average_loss /= self.number_of_steps
        print("Average Loss: "+str(average_loss))
        return average_loss

    def _tf_params_to_numpy(self):
        new_param = dict()
        for var in tf.trainable_variables():
            new_param[var.name[:-2]] = var.eval(self.mon_sess)
        return new_param
