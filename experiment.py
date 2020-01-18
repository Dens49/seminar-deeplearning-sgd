import torch


class Experiment:
    def __init__(self, name, optimizer_name, params, lr_scheduler_name=""):
        self.name = name
        self.params = params
        self.optimizer_name = optimizer_name
        self.optimizer = None
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_scheduler = None

    def has_optimizer(self):
        return self.optimizer is not None

    def has_lr_scheduler(self):
        return self.lr_scheduler is not None

    def create_optimizer(self, model_parameters):
        self.optimizer = None
        if self.optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD(model_parameters,
                                             lr=self.params["lr"])
        elif self.optimizer_name == "SGD_WITH_MOMENTUM":
            self.optimizer = torch.optim.SGD(model_parameters,
                                             lr=self.params["lr"],
                                             momentum=self.params["momentum"])
        elif self.optimizer_name == "Adagrad":
            self.optimizer = torch.optim.Adagrad(model_parameters)
        elif self.optimizer_name == "Rmsprop":
            self.optimizer = torch.optim.RMSprop(model_parameters,
                                                 lr=self.params["lr"],
                                                 momentum=self.params["momentum"])
        elif self.optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(model_parameters,
                                              lr=self.params["lr"],
                                              betas=self.params["betas"],
                                              eps=self.params["eps"])

        if self.lr_scheduler_name == "COSINE_ANNEALING_WARM_RESTARTS":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=self.params["T_0"], T_mult=self.params["T_mult"])
        elif self.lr_scheduler_name == "STEP_LR":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                step_size=self.params["lr_step_size"],
                                                                gamma=self.params["lr_gamma"])
        return self.optimizer
