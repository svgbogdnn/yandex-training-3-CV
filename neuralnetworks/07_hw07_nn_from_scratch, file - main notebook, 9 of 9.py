import numpy as np

# 1 Module is an abstract class which defines fundamental methods necessary for a training a neural network. You do not need to change anything here, just read the comments.

class Module(object):
    """
    Basically, you can think of a module as of a something (black box)
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`:

        output = module.forward(input)

    The module should be able to perform a backward pass: to differentiate the `forward` function.
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule.

        gradInput = module.backward(input, gradOutput)
    """

    def __init__(self):
        self.output = None
        self.gradInput = None
        self.training = True
        # параметры и их градиенты (будут заполняться в наследниках)
        self._parameters = []
        self._gradParameters = []

    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        return self.updateOutput(input)

    def backward(self, input, gradOutput):
        """
        Performs a backpropagation step through the module, with respect to the given input.

        This includes
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput

    def updateOutput(self, input):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the `output` field.

        Make sure to both store the data in `output` field and return it.
        """
        # identity implementation
        self.output = input
        return self.output

    def updateGradInput(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own input.
        This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.

        The shape of `gradInput` is always the same as the shape of `input`.

        Make sure to both store the gradients in `gradInput` field and return it.
        """
        # identity implementation
        self.gradInput = gradOutput
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        # базовый модуль не имеет параметров
        return

    def zeroGradParameters(self):
        """
        Zeroes `gradParams` variable if the module has params.
        """
        for gp in self._gradParameters:
            gp[...] = 0

    def getParameters(self):
        """
        Returns a list with its parameters.
        If the module does not have parameters return empty list.
        """
        return list(self._parameters)

    def getGradParameters(self):
        """
        Returns a list with gradients with respect to its parameters.
        If the module does not have parameters return empty list.
        """
        return list(self._gradParameters)

    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True

    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout and BatchNorm.
        """
        self.training = False

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        cls = self.__class__.__name__
        mode = "train" if self.training else "eval"
        return f"<{cls} ({mode})>"

# 2 Sequential container, that define a forward and backward pass procedures

class Sequential(Module):
    """
         This class implements a container, which processes `input` data sequentially.

         `input` is processed by each module (layer) in self.modules consecutively.
         The resulting array is called `output`.
    """

    def __init__(self):
        super(Sequential, self).__init__()
        self.modules = []
        # cache inputs to each layer during forward
        self._layer_inputs = []

    def add(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def updateOutput(self, input):
        """
        Basic workflow of FORWARD PASS:

            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})

        Just write a little loop.
        """
        # clear any previous cache
        self._layer_inputs = []
        current = input
        for module in self.modules:
            # remember what this module saw
            self._layer_inputs.append(current)
            # forward through it
            current = module.forward(current)
        self.output = current
        return self.output

    def backward(self, input, gradOutput):
        """
        Workflow of BACKWARD PASS:

            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)
            gradInput = module[0].backward(input, g_1)

        To each module you need to provide the input it saw during the forward pass.
        """
        grad = gradOutput
        # traverse modules in reverse, paired with their cached inputs
        for module, layer_input in zip(reversed(self.modules),
                                       reversed(self._layer_inputs)):
            grad = module.backward(layer_input, grad)
        self.gradInput = grad
        return self.gradInput

    def zeroGradParameters(self):
        for module in self.modules:
            module.zeroGradParameters()

    def getParameters(self):
        """
        Should gather all parameters in a list.
        """
        params = []
        for module in self.modules:
            params.extend(module.getParameters())
        return params

    def getGradParameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        grads = []
        for module in self.modules:
            grads.extend(module.getGradParameters())
        return grads

    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string

    def __getitem__(self, x):
        return self.modules[x]

    def train(self):
        """
        Propagates training parameter through all modules
        """
        self.training = True
        for module in self.modules:
            module.train()

    def evaluate(self):
        """
        Propagates evaluation parameter through all modules
        """
        self.training = False
        for module in self.modules:
            module.evaluate()


''' LAYERS '''

# 3 Linear transform layer

class Linear(Module):
    """
    A module which applies a linear transformation
    A common name is fully-connected layer, InnerProductLayer in caffe.

    The module should work with 2D input of shape (n_samples, n_feature).
    """

    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()

        # This is a nice initialization
        stdv = 1. / np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size=(n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size=(n_out,))

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        # input: (batch_size, n_in)
        # output: (batch_size, n_out)
        # compute affine transform
        #   output_i = W x_i + b
        self.output = input.dot(self.W.T) + self.b  # broadcasting b
        return self.output

    def updateGradInput(self, input, gradOutput):
        # gradOutput: (batch_size, n_out)
        # gradInput: (batch_size, n_in) = gradOutput . W
        self.gradInput = gradOutput.dot(self.W)
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        # accumulate gradients w.r.t. W and b
        # dL/dW = gradOutput^T . input
        # dL/db = sum over batch of gradOutput
        self.gradW += gradOutput.T.dot(input)
        self.gradb += gradOutput.sum(axis=0)

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        n_in = self.W.shape[1]
        n_out = self.W.shape[0]
        return f"Linear {n_in} -> {n_out}"

# 4 SoftMax

class SoftMax(Module):
    def __init__(self):
        super(SoftMax, self).__init__()

    def updateOutput(self, input):
        # subtract max for numerical stability (per row)
        shifted = input - input.max(axis=1, keepdims=True)
        exps = np.exp(shifted)
        sums = exps.sum(axis=1, keepdims=True)
        self.output = exps / sums
        return self.output

    def updateGradInput(self, input, gradOutput):
        # gradOutput: shape (batch, n_feats)
        # self.output (softmax probabilities): same shape
        p = self.output
        # dot = sum_k gradOutput_k * p_k per row
        dot = (gradOutput * p).sum(axis=1, keepdims=True)
        # dL/dinput_j = p_j * (gradOutput_j - dot)
        self.gradInput = p * (gradOutput - dot)
        return self.gradInput

    def __repr__(self):
        return "SoftMax"

# 5 LogSoftMax

class LogSoftMax(Module):
    def __init__(self):
        super(LogSoftMax, self).__init__()

    def updateOutput(self, input):
        # subtract max for numerical stability
        shifted = input - input.max(axis=1, keepdims=True)
        exps = np.exp(shifted)
        sums = exps.sum(axis=1, keepdims=True)
        # log softmax: log(exp(shifted) / sums) = shifted - log(sums)
        self.output = shifted - np.log(sums)
        return self.output

    def updateGradInput(self, input, gradOutput):
        # gradOutput: (batch, n_feats)
        # p = softmax = exp(logsoftmax)
        p = np.exp(self.output)
        # sum of gradOutput over features
        sum_grad = gradOutput.sum(axis=1, keepdims=True)
        # gradInput = gradOutput - p * sum_grad
        self.gradInput = gradOutput - p * sum_grad
        return self.gradInput

    def __repr__(self):
        return "LogSoftMax"

# 6 Batch normalization

class BatchNormalization(Module):
    EPS = 1e-3

    def __init__(self, alpha=0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = None
        self.moving_variance = None
        # placeholders for backward
        self._batch_mean = None
        self._batch_var = None
        self._x_centered = None
        self._std = None

    def updateOutput(self, input):
        # input: (batch_size, n_feats)
        if self.training:
            # compute batch statistics
            batch_mean = input.mean(axis=0)
            batch_var = ((input - batch_mean) ** 2).mean(axis=0)
            # update moving averages
            if self.moving_mean is None:
                self.moving_mean = batch_mean.copy()
                self.moving_variance = batch_var.copy()
            else:
                self.moving_mean = (
                        self.moving_mean * self.alpha
                        + batch_mean * (1 - self.alpha)
                )
                self.moving_variance = (
                        self.moving_variance * self.alpha
                        + batch_var * (1 - self.alpha)
                )
            mean = batch_mean
            var = batch_var
        else:
            # use stored moving averages
            mean = self.moving_mean
            var = self.moving_variance

        # normalize
        x_centered = input - mean
        std = np.sqrt(var + self.EPS)
        output = x_centered / std

        # cache for backward
        if self.training:
            self._batch_mean = mean
            self._batch_var = var
            self._x_centered = x_centered
            self._std = std

        self.output = output
        return self.output

    def updateGradInput(self, input, gradOutput):
        # only valid in training mode (otherwise gradInput = gradOutput / std_moving)
        if not self.training:
            # during eval, just normalize by moving stats
            std = np.sqrt(self.moving_variance + self.EPS)
            self.gradInput = gradOutput / std
            return self.gradInput

        # training backward
        batch_size = input.shape[0]
        x_c = self._x_centered
        std = self._std

        # gradient through normalization
        dxhat = gradOutput  # shape (B, F)
        # dvar
        dvar = np.sum(
            dxhat * x_c * -0.5 * std ** (-3),
            axis=0
        )
        # dmean
        dmean = np.sum(
            -dxhat / std,
            axis=0
        ) + dvar * np.mean(-2.0 * x_c, axis=0)

        # final gradient w.r.t. input
        self.gradInput = (
                dxhat / std
                + dvar * 2.0 * x_c / batch_size
                + dmean / batch_size
        )
        return self.gradInput

    def __repr__(self):
        return "BatchNormalization"

# 7 Dropout

class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p
        self.mask = None

    def updateOutput(self, input):
        if self.training:
            # sample mask: 1 with prob 1-p, 0 with prob p
            self.mask = (np.random.rand(*input.shape) >= self.p).astype(input.dtype)
            # scale to keep expectation
            self.output = input * self.mask / (1.0 - self.p)
        else:
            # evaluation: identity
            self.output = input
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.training:
            # backprop through same mask and scale
            self.gradInput = gradOutput * self.mask / (1.0 - self.p)
        else:
            # evaluation: identity
            self.gradInput = gradOutput
        return self.gradInput

    def __repr__(self):
        return "Dropout"

# 8 Activation functions - ReLU, Leaky ReLU, ELU, SoftPlus 8-11

class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput, input > 0)
        return self.gradInput

    def __repr__(self):
        return "ReLU"

class LeakyReLU(Module):
    def __init__(self, slope=0.03):
        super(LeakyReLU, self).__init__()
        self.slope = slope

    def updateOutput(self, input):
        # Leaky version: positive branch same, negative scaled
        self.output = np.where(input > 0, input, self.slope * input)
        return self.output

    def updateGradInput(self, input, gradOutput):
        # gradient is 1 for input>0, slope for input<=0
        grad = np.where(input > 0, 1.0, self.slope)
        self.gradInput = gradOutput * grad
        return self.gradInput

    def __repr__(self):
        return "LeakyReLU"

class ELU(Module):
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()
        self.alpha = alpha

    def updateOutput(self, input):
        # ELU: x if x>0 else alpha*(exp(x)-1)
        positive = np.maximum(input, 0)
        negative = self.alpha * (np.exp(np.minimum(input, 0)) - 1)
        self.output = positive + negative
        return self.output

    def updateGradInput(self, input, gradOutput):
        # derivative is 1 for x>0, else output + alpha over negative branch
        mask = input > 0
        # for negative, d/dx alpha*(exp(x)-1) = alpha*exp(x) = output + alpha
        grad_neg = (self.output + self.alpha)
        grad = np.where(mask, 1.0, grad_neg)
        self.gradInput = gradOutput * grad
        return self.gradInput

    def __repr__(self):
        return "ELU"

class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()

    def updateOutput(self, input):
        # SoftPlus: log(1 + exp(x))
        self.output = np.log1p(np.exp(input))
        return self.output

    def updateGradInput(self, input, gradOutput):
        # derivative is sigmoid(x) = 1/(1+exp(-x))
        sigmoid = 1.0 / (1.0 + np.exp(-input))
        self.gradInput = gradOutput * sigmoid
        return self.gradInput

    def __repr__(self):
        return "SoftPlus"

# 12 Criterions

class Criterion(object):
    def __init__(self):
        self.output = None
        self.gradInput = None

    def forward(self, input, target):
        """
            Given an input and a target, compute the loss function
            associated to the criterion and return the result.

            For consistency this function should not be overrided,
            all the code goes in `updateOutput`.
        """
        return self.updateOutput(input, target)

    def backward(self, input, target):
        """
            Given an input and a target, compute the gradients of the loss function
            associated to the criterion and return the result.

            For consistency this function should not be overrided,
            all the code goes in `updateGradInput`.
        """
        return self.updateGradInput(input, target)

    def updateOutput(self, input, target):
        """
        Function to override.
        """
        return self.output

    def updateGradInput(self, input, target):
        """
        Function to override.
        """
        return self.gradInput

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Criterion"

# 13 MSECriterion

class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()

    def updateOutput(self, input, target):
        self.output = np.sum(np.power(input - target, 2)) / input.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        self.gradInput = (input - target) * 2 / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "MSECriterion"

# 14 Negative LogLikelihood criterion (numerically unstable)

class ClassNLLCriterionUnstable(Criterion):
    EPS = 1e-15

    def __init__(self):
        super(ClassNLLCriterionUnstable, self).__init__()

    def updateOutput(self, input, target):
        # clamp to avoid log(0)
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        # only nonzero where target=1; sum over features then average over batch
        batch_size = input.shape[0]
        # elementwise target * log probability
        losses = - np.sum(target * np.log(input_clamp), axis=1)
        self.output = np.mean(losses)
        return self.output

    def updateGradInput(self, input, target):
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        batch_size = input.shape[0]
        # gradient of -mean(target * log(p)) wrt p is - target / (p * N)
        self.gradInput = - target / (input_clamp * batch_size)
        return self.gradInput

    def __repr__(self):
        return "ClassNLLCriterionUnstable"

# 15 Negative LogLikelihood criterion (numerically stable)

class ClassNLLCriterion(Criterion):
    def __init__(self):
        super(ClassNLLCriterion, self).__init__()

    def updateOutput(self, input, target):
        # input is log-probabilities
        # loss = -mean over batch of sum(target * log-prob)
        batch_size = input.shape[0]
        losses = - np.sum(target * input, axis=1)
        self.output = np.mean(losses)
        return self.output

    def updateGradInput(self, input, target):
        # gradient of -mean(target * input) w.r.t. input is - target / batch_size
        batch_size = input.shape[0]
        self.gradInput = - target / batch_size
        return self.gradInput

    def __repr__(self):
        return "ClassNLLCriterion"

# Optimizers

# 16 SGD optimizer with momentum

def sgd_momentum(variables, gradients, config, state):
    """
    Универсальный SGD с моментумом.

    Аргументы:
        variables: список списков параметров (по одному списку на модуль),
        gradients: список списков соответствующих градиентов,
        config: {"learning_rate": float, "momentum": float},
        state: словарь для хранения состояния (накопленных моментов).

    Механика:
        v = momentum * v + lr * grad
        param -= v
    """
    # храним накопленные моменты в state['accumulated_grads']
    acc = state.setdefault('accumulated_grads', {})

    var_index = 0
    for vars_layer, grads_layer in zip(variables, gradients):
        for param, grad in zip(vars_layer, grads_layer):
            # инициализируем нулевой момент, если нет
            v = acc.setdefault(var_index, np.zeros_like(grad))
            # v = momentum * v + lr * grad
            np.add(config['momentum'] * v, config['learning_rate'] * grad, out=v)
            # обновляем параметр
            param -= v
            var_index += 1

# 17 Adam Optimizer

def adam_optimizer(variables, gradients, config, state):
    state.setdefault('m', {})  # first moments
    state.setdefault('v', {})  # second moments
    state.setdefault('t', 0)   # timestep
    state['t'] += 1
    for k in ['learning_rate', 'beta1', 'beta2', 'epsilon']:
        assert k in config, config.keys()

    var_index = 0
    # bias-corrected learning rate
    lr_t = config['learning_rate'] * np.sqrt(1 - config['beta2']**state['t']) / (1 - config['beta1']**state['t'])
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            m = state['m'].setdefault(var_index, np.zeros_like(current_grad))
            v = state['v'].setdefault(var_index, np.zeros_like(current_grad))

            # 1) update biased first moment estimate
            #    m = beta1 * m + (1-beta1) * grad
            np.add(config['beta1'] * m,
                   (1 - config['beta1']) * current_grad,
                   out=m)

            # 2) update biased second raw moment estimate
            #    v = beta2 * v + (1-beta2) * grad^2
            np.add(config['beta2'] * v,
                   (1 - config['beta2']) * (current_grad ** 2),
                   out=v)

            # 3) update parameters
            #    param -= lr_t * m / (sqrt(v) + epsilon)
            current_var -= lr_t * m / (np.sqrt(v) + config['epsilon'])

            # sanity check: in-place updates
            assert m is state['m'][var_index]
            assert v is state['v'][var_index]

            var_index += 1

# 18 Conv2d [Advanced]

import numpy as np
import scipy.signal

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2d, self).__init__()
        assert kernel_size % 2 == 1, kernel_size

        stdv = 1. / np.sqrt(in_channels)
        self.W = np.random.uniform(-stdv, stdv,
                                   size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.random.uniform(-stdv, stdv, size=(out_channels,))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        # input: batch x in_ch x h x w
        batch_size, _, h, w = input.shape
        pad = self.kernel_size // 2
        # сохраняем padded input для backward
        self._input_padded = np.pad(input,
                                    ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                                    mode='constant', constant_values=0)
        out_h, out_w = h, w
        self.output = np.zeros((batch_size, self.out_channels, out_h, out_w), dtype=input.dtype)

        for n in range(batch_size):
            for oc in range(self.out_channels):
                # корреляция каждой карты признаков входа
                accum = np.zeros((out_h, out_w), dtype=input.dtype)
                for ic in range(self.in_channels):
                    accum += scipy.signal.correlate(
                        self._input_padded[n, ic],
                        self.W[oc, ic],
                        mode='valid'
                    )
                self.output[n, oc] = accum + self.b[oc]
        return self.output

    def updateGradInput(self, input, gradOutput):
        # gradOutput: batch x out_ch x h x w
        batch_size, _, h, w = input.shape
        pad = self.kernel_size // 2
        # нам нужен тот же padded input, но тут мы будем коррелировать
        # градиенты вывода с весами, чтобы получить gradInput
        self.gradInput = np.zeros_like(self._input_padded)

        for n in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    # "valid" корреляция gradOutput[n,oc] с W[oc,ic] даст вклад в gradInput[n,ic]
                    self.gradInput[n, ic] += scipy.signal.correlate(
                        gradOutput[n, oc],
                        self.W[oc, ic],
                        mode='full'  # full, чтобы восстановить padding
                    )
        # убрать padding
        self.gradInput = self.gradInput[:, :, pad:pad+h, pad:pad+w]
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        # накопить в self.gradW и self.gradb
        batch_size, _, h, w = input.shape
        pad = self.kernel_size // 2
        # используем тот же padded вход
        for oc in range(self.out_channels):
            # bias: сумма по всем элементам gradOutput[:,oc]
            self.gradb[oc] += gradOutput[:, oc].sum()
            for ic in range(self.in_channels):
                accum = np.zeros_like(self.W[oc, ic])
                for n in range(batch_size):
                    # корреляция padded input с gradOutput:
                    accum += scipy.signal.correlate(
                        self._input_padded[n, ic],
                        gradOutput[n, oc],
                        mode='valid'
                    )
                self.gradW[oc, ic] += accum
        # теперь градиенты накоплены в self.gradW, self.gradb

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        return f'Conv2d {s[1]}->{s[0]} ({s[2]}×{s[3]})'

# 19 MaxPool2d [Advanced]

import numpy as np

class MaxPool2d(Module):
    def __init__(self, kernel_size):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.gradInput = None
        self.max_indices = None

    def updateOutput(self, input):
        # input: (batch, channels, h, w)
        batch, channels, h, w = input.shape
        k = self.kernel_size
        assert h % k == 0 and w % k == 0, "For simplicity, require h%k==0 and w%k==0"
        oh, ow = h // k, w // k

        # reshape + transpose -> (batch, ch, oh, ow, k, k)
        x = input.reshape(batch, channels, oh, k, ow, k)
        x = x.transpose(0, 1, 2, 4, 3, 5)

        # flatten the last two dims: shape -> (batch, ch, oh, ow, k*k)
        flat = x.reshape(batch, channels, oh, ow, k * k)

        # для каждого окна находим индекс максимума
        self.max_indices = np.argmax(flat, axis=-1)  # (batch, ch, oh, ow)
        # извлекаем сами максимумы
        self.output = np.max(flat, axis=-1)           # (batch, ch, oh, ow)

        return self.output

    def updateGradInput(self, input, gradOutput):
        # gradOutput: (batch, ch, oh, ow)
        batch, channels, h, w = input.shape
        k = self.kernel_size
        oh, ow = h // k, w // k

        # подготовим нулевой градиент того же размера, что input
        self.gradInput = np.zeros_like(input)

        # расставляем по индексам сохранённые градиенты
        for n in range(batch):
            for c in range(channels):
                for i in range(oh):
                    for j in range(ow):
                        idx = self.max_indices[n, c, i, j]
                        di, dj = divmod(idx, k)
                        self.gradInput[n, c, i * k + di, j * k + dj] = gradOutput[n, c, i, j]

        return self.gradInput

    def __repr__(self):
        return f"MaxPool2d(kern={self.kernel_size}, stride={self.kernel_size})"

# 20 Flatten Layer

class Flatten(Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def updateOutput(self, input):
        self.output = input.reshape(len(input), -1)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput.reshape(input.shape)
        return self.gradInput

    def __repr__(self):
        return "Flatten"

# 21 Extra layer

class ChannelwiseScaling(Module):
    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()
        stdv = 1./np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta  = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta  = np.zeros_like(self.beta)

    def updateOutput(self, input):
        # для каждого элемента batch и каждого канала:
        # output[:, c] = input[:, c] * gamma[c] + beta[c]
        self.output = input * self.gamma + self.beta
        return self.output

    def updateGradInput(self, input, gradOutput):
        # gradInput = gradOutput * gamma
        self.gradInput = gradOutput * self.gamma
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        # градиенты по beta — просто сумма дельт по батчу
        self.gradBeta  = np.sum(gradOutput, axis=0)
        # градиенты по gamma — сумма input*gradOutput по батчу
        self.gradGamma = np.sum(gradOutput * input, axis=0)

    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)

    def getParameters(self):
        return [self.gamma, self.beta]

    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]

    def __repr__(self):
        return "ChannelwiseScaling"








"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
import sys
import os
import time
import numpy as np
import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

def load_dataset():
    """
    Возвращает кортежи (X_train, y_train, X_val, y_val, X_test, y_test),
    где X_*.shape = (N, 28, 28), y_* — 1D numpy-массивы меток.
    Разбиваем train→train+val по last 10k образцов.
    """
    # 1) загружаем
    train_ds = FashionMNIST(".", train=True,  transform=ToTensor(), download=True)
    test_ds  = FashionMNIST(".", train=False, transform=ToTensor(), download=True)

    # 2) извлекаем numpy-массивы и нормируем [0,1]
    X_full = train_ds.data.numpy().astype(np.float32) / 255.0  # shape (60000,28,28)
    y_full = train_ds.targets.numpy()                          # shape (60000,)

    X_test = test_ds.data.numpy().astype(np.float32) / 255.0   # shape (10000,28,28)
    y_test = test_ds.targets.numpy()                           # shape (10000,)

    # 3) разбиваем train→train/val
    split = 50000
    X_train, y_train = X_full[:split], y_full[:split]
    X_val,   y_val   = X_full[split:], y_full[split:]

    return X_train, y_train, X_val, y_val, X_test, y_test

def get_accuracy(model, loader):
    """
    model — ваша сеть, у которой есть .forward()
    loader — DataLoader, отдающий пары (X_batch, y_batch) в numpy-формате:
             X_batch.shape = (B, ...), y_batch.shape = (B, num_classes)
    """
    model.evaluate()   # переводим в режим inference, если у вас есть различие train/eval
    correct = 0
    total = 0
    for Xb, yb in loader:
        # если Xb, yb в torch.Tensor — конвертируйте в numpy:
        if hasattr(Xb, 'cpu'):
            Xb_np = Xb.cpu().numpy()
        else:
            Xb_np = Xb
        out = model.forward(Xb_np)       # shape (B, num_classes)
        preds = np.argmax(out, axis=1)   # ваши предсказанные классы
        trues = np.argmax(yb, axis=1)    # ваши истинные (one-hot → class idx)
        correct += (preds == trues).sum()
        total += len(trues)
    return correct / total











"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

# Generate some data
N = 500

X1 = np.random.randn(N,2) + np.array([2,2])
X2 = np.random.randn(N,2) + np.array([-2,-2])

Y = np.concatenate([np.ones(N),np.zeros(N)])[:,None]
Y = np.hstack([Y, 1-Y])

X = np.vstack([X1,X2])
plt.scatter(X[:,0],X[:,1], c = Y[:,0], edgecolors= 'none')

net = Sequential()
net.add(Linear(2, 2))
net.add(LogSoftMax())

criterion = ClassNLLCriterion()

print(net)

# Iptimizer params
optimizer_config = {'learning_rate' : 1e-1, 'momentum': 0.9}
optimizer_state = {}

# Looping params
n_epoch = 20
batch_size = 128


# batch generator
def get_batches(dataset, batch_size):
    X, Y = dataset
    n_samples = X.shape[0]

    # Shuffle at the start of epoch
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)

        batch_idx = indices[start:end]

        yield X[batch_idx], Y[batch_idx]


loss_history = []

for i in range(n_epoch):
    for x_batch, y_batch in get_batches((X, Y), batch_size):
        net.zeroGradParameters()

        # Forward
        predictions = net.forward(x_batch)
        loss = criterion.forward(predictions, y_batch)

        # Backward
        dp = criterion.backward(predictions, y_batch)
        net.backward(x_batch, dp)

        # Update weights
        sgd_momentum(net.getParameters(),
                     net.getGradParameters(),
                     optimizer_config,
                     optimizer_state)

        loss_history.append(loss)

    print('Current loss: %f' % loss)

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()


num_classes = 10

def to_one_hot(y, num_classes):
    """
    y: 1D array of integer labels, shape (N,)
    returns: 2D array, shape (N, num_classes)
    """
    eye = np.eye(num_classes, dtype=np.float32)
    return eye[y]

y_train = to_one_hot(y_train, num_classes)
y_val   = to_one_hot(y_val,   num_classes)
y_test  = to_one_hot(y_test,  num_classes)

# Архитектура: Linear → [BatchNorm → ChannelwiseScaling] → Activation → Linear → LogSoftMax
def build_net(activation_cls, use_bn=False):
    net = Sequential()
    net.add(Linear(784, 256))
    if use_bn:
        net.add(BatchNormalization(alpha=0.9))
        net.add(ChannelwiseScaling(256))
    net.add(activation_cls())
    net.add(Linear(256, 10))
    net.add(LogSoftMax())
    return net

# Общая функция обучения, возвращает историю потерь (список)
def train_net(net, optimizer_fn, optimizer_cfg, n_epoch=10, batch_size=128):
    optimizer_state = {}
    loss_hist = []
    criterion = ClassNLLCriterion()
    for epoch in range(n_epoch):
        for x_batch, y_batch in get_batches((X_train, y_train), batch_size):
            net.zeroGradParameters()
            preds = net.forward(x_batch.reshape(-1,784))
            loss = criterion.forward(preds, y_batch)
            dp = criterion.backward(preds, y_batch)
            net.backward(x_batch.reshape(-1,784), dp)
            optimizer_fn(net.getParameters(),
                         net.getGradParameters(),
                         optimizer_cfg,
                         optimizer_state)
            loss_hist.append(loss)
    return loss_hist

# Параметры эксперимента
activations = [ReLU, LeakyReLU, ELU, SoftPlus]
labels = ['ReLU','LeakyReLU','ELU','SoftPlus']
optimizers = {
    'SGD': (sgd_momentum, {'learning_rate':1e-1, 'momentum':0.9}),
    'Adam': (adam_optimizer, {'learning_rate':1e-3, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-8})
}


for i, act in enumerate(activations):
    for j, use_bn in enumerate([False, True]):
        for opt_name, (opt_fn, opt_cfg) in optimizers.items():
            net = build_net(act, use_bn=use_bn)
            net.train()
            tag = f"{labels[i]}{' +BN' if use_bn else ''} ({opt_name})"
            print("Training", tag)
            lh = train_net(net, opt_fn, opt_cfg, n_epoch=5, batch_size=256)
            # строим кривую в лог-координатах
            plt.plot(np.log(lh), label=tag)

plt.title("Сравнение активаций (логарифм потерь)")
plt.xlabel("Итерации")
plt.ylabel("log(loss)")
plt.legend(fontsize='small', ncol=2)
# plt.grid(True)
# plt.show()

import numpy as np
from tqdm import tqdm

# гиперпараметры
n_epochs = 10
batch_size = 128
hidden_units = 256
lr_sgd = 1e-2
lr_adam = 1e-3
momentum = 0.9

activations = {
    'ReLU': ReLU,
    'LeakyReLU': lambda: LeakyReLU(0.03),
    'ELU': lambda: ELU(1.0),
    'SoftPlus': SoftPlus
}

def make_net(act_cls, use_bn=False):
    net = Sequential()
    net.add(Flatten())
    net.add(Linear(28*28, hidden_units))
    if use_bn:
        net.add(BatchNormalization())
        net.add(ChannelwiseScaling(hidden_units))
    net.add(act_cls())
    net.add(Linear(hidden_units, 10))
    net.add(LogSoftMax())
    return net

def train_net(net, optimizer_fn, opt_config):
    loss_hist = []
    optimizer_state = {}
    for epoch in range(n_epochs):
        for Xb, yb in get_batches((X_train, y_train), batch_size):
            net.zeroGradParameters()
            out = net.forward(Xb)
            loss = criterion.forward(out, yb)
            dp = criterion.backward(out, yb)
            net.backward(Xb, dp)
            optimizer_fn(net.getParameters(),
                         net.getGradParameters(),
                         opt_config,
                         optimizer_state)
            loss_hist.append(loss)
    return loss_hist

# загрузим и one-hot закодируем MNIST
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
y_train = to_one_hot(y_train, 10)
y_val   = to_one_hot(y_val,   10)

criterion = ClassNLLCriterion()

results = {}

results = {}

for name, act_cls in activations.items():
    for use_bn in (False, True):
        tag = f"{name}" + ("+BN" if use_bn else "")

        # --- SGD ---
        net = make_net(act_cls, use_bn)   # создаём сеть
        net.train()                       # переводим в режим train (не присваивая!)
        loss_sgd = train_net(
            net,
            sgd_momentum,
            {'learning_rate': lr_sgd, 'momentum': momentum}
        )

        # --- Adam ---
        net = make_net(act_cls, use_bn)   # снова создаём чистую сеть
        net.train()                       # переводим в режим train
        loss_adam = train_net(
            net,
            adam_optimizer,
            {'learning_rate': lr_adam,
             'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}
        )

        results[tag] = {'SGD': loss_sgd, 'Adam': loss_adam}

# Final “super cool” MNIST model with Dropout, BatchNorm, LR decay and simple augmentation
import numpy as np
from tqdm import tqdm

# --- Hyperparameters ---
n_epochs = 20
batch_size = 128
lr_initial = 1e-3
momentum = 0.9
hidden1 = 512
hidden2 = 256
drop_p = 0.5

# Simple augmentation: random shifts up to ±2 pixels
def augment_batch(X, max_shift=2):
    N, C, H, W = X.shape
    X_aug = np.zeros_like(X)
    for i in range(N):
        dx = np.random.randint(-max_shift, max_shift+1)
        dy = np.random.randint(-max_shift, max_shift+1)
        src = X[i, 0]  # single channel
        dst = np.zeros_like(src)
        # compute valid box
        x0, x1 = max(0, dx), min(W, W+dx)
        y0, y1 = max(0, dy), min(H, H+dy)
        sx0, sx1 = max(0, -dx), min(W, W-dx)
        sy0, sy1 = max(0, -dy), min(H, H-dy)
        dst[sy0:sy1, sx0:sx1] = src[y0:y1, x0:x1]
        X_aug[i,0] = dst
    return X_aug

# Build the network
def make_super_net():
    net = Sequential()
    net.add(Flatten())
    net.add(Linear(28*28, hidden1))
    net.add(BatchNormalization())
    net.add(ChannelwiseScaling(hidden1))
    net.add(ReLU())
    net.add(Dropout(drop_p))
    net.add(Linear(hidden1, hidden2))
    net.add(BatchNormalization())
    net.add(ChannelwiseScaling(hidden2))
    net.add(ReLU())
    net.add(Dropout(drop_p))
    net.add(Linear(hidden2, 10))
    net.add(LogSoftMax())
    return net

# Prepare data (shape X_train: (N,1,28,28), y_train one-hot)
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
X_train = X_train.reshape(-1,1,28,28)
X_val   = X_val.reshape(-1,1,28,28)
y_train = to_one_hot(y_train, 10)
y_val   = to_one_hot(y_val,   10)

criterion = ClassNLLCriterion()

# Instantiate model, optimizer config, scheduler
net = make_super_net()
net.train()
opt_config = {'learning_rate': lr_initial, 'momentum': momentum}
scheduler_step = 5
scheduler_gamma = 0.5
optimizer_state = {}

# Training loop
loss_history = []
for epoch in range(1, n_epochs+1):
    pbar = tqdm(get_batches((X_train, y_train), batch_size),
                desc=f"Epoch {epoch}/{n_epochs}", unit="batch")
    for Xb, yb in pbar:
        net.zeroGradParameters()
        Xb_aug = augment_batch(Xb, max_shift=2)
        out = net.forward(Xb_aug)
        loss = criterion.forward(out, yb)
        dp = criterion.backward(out, yb)
        net.backward(Xb_aug, dp)
        sgd_momentum(net.getParameters(),
                     net.getGradParameters(),
                     opt_config,
                     optimizer_state)
        loss_history.append(loss)
        pbar.set_postfix(loss=f"{loss:.4f}")
    # LR decay
    if epoch % scheduler_step == 0:
        opt_config['learning_rate'] *= scheduler_gamma


# --- Утилиты для оценки и батчей (без использования PyTorch) ---
def get_accuracy(net, dataset, batch_size=128):
    """
    net: ваша модель Sequential
    dataset: list of (X, y) пар, где
        X — массив shape (28*28,) или (28,28) для одного примера
        y — one-hot вектор shape (10,)
    """
    correct = 0
    total = len(dataset)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = dataset[start:end]
        Xb = np.stack([x.reshape(-1) for x, _ in batch], axis=0)
        yb = np.stack([y for _, y in batch], axis=0)
        out = net.forward(Xb)
        preds = np.argmax(out, axis=1)
        trues = np.argmax(yb, axis=1)
        correct += np.sum(preds == trues)
    return correct / total

# --- Пример использования после обучения и перключения в режим evaluate() ---

# Предполагаем, что X_train, y_train, X_val, y_val, X_test, y_test уже есть и one-hot

net.evaluate()

train_dataset = list(zip(X_train, y_train))
val_dataset   = list(zip(X_val,   y_val))
test_dataset  = list(zip(X_test,  y_test))

batch_size = 128

train_acc = get_accuracy(net, train_dataset, batch_size)
val_acc   = get_accuracy(net, val_dataset,   batch_size)
print(f"Train accuracy: {train_acc:.4f},  Val accuracy: {val_acc:.4f}")

from torch.utils.data import DataLoader
net.evaluate()
test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False)
test_acc = get_accuracy(net, test_loader)
print(f"Test accuracy: {test_acc:.4f}")



# --- Оценка точности на тестовом наборе ---

# Предполагаем, что у вас уже загружены и one-hot закодированы:
# X_test — массив формы (N_test, 28, 28),
# y_test — массив формы (N_test, 10)

# Если нужно, переформатируем входы в векторы:
X_test_flat = X_test.reshape(len(X_test), -1)

# Задаём гиперпараметры
batch_size = 128

# Функция для генерации батчей
def get_batches(X, Y, batch_size):
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = indices[start:end]
        yield X[idx], Y[idx]

# Создаём и тренируем «крутую» модель (пример)
net = Sequential()
net.add(Flatten())
net.add(Linear(28*28, 256))
net.add(BatchNormalization())
net.add(ChannelwiseScaling(256))
net.add(ReLU())
net.add(Dropout(0.5))
net.add(Linear(256, 10))
net.add(LogSoftMax())

criterion = ClassNLLCriterion()
optim_config = {'learning_rate': 1e-2, 'momentum': 0.9}
optim_state = {}

# Обучение
epochs = 10
for epoch in range(epochs):
    for Xb, yb in get_batches(X_train.reshape(len(X_train), -1), y_train, batch_size):
        net.zeroGradParameters()
        out = net.forward(Xb)
        loss = criterion.forward(out, yb)
        dp = criterion.backward(out, yb)
        net.backward(Xb, dp)
        sgd_momentum(net.getParameters(), net.getGradParameters(), optim_config, optim_state)
    print(f"Epoch {epoch+1}/{epochs} completed")

# Переключаем модель в режим оценки (выключаем Dropout, BatchNorm)
net.evaluate()

# Подсчёт точности на тесте
correct = 0
total = X_test_flat.shape[0]
for Xb, yb in get_batches(X_test_flat, y_test, batch_size):
    out = net.forward(Xb)
    preds = np.argmax(out, axis=1)
    trues = np.argmax(yb, axis=1)
    correct += np.sum(preds == trues)

accuracy = correct / total
print(f"Test accuracy: {accuracy * 100:.2f}%")  # ожидаем около 90%




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Предполагаем, что X_train, y_train, X_val, y_val, X_test, y_test уже есть в виде numpy-массивов,
# y_* — one-hot. Переведем их в тензоры и создадим датасеты:

X_train_t = torch.FloatTensor(X_train)           # shape (N_train, 784)
y_train_t = torch.LongTensor(np.argmax(y_train, axis=1))  # shape (N_train,)

X_val_t   = torch.FloatTensor(X_val)
y_val_t   = torch.LongTensor(np.argmax(y_val,   axis=1))

X_test_t  = torch.FloatTensor(X_test)
y_test_t  = torch.LongTensor(np.argmax(y_test,  axis=1))

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t,   y_val_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)

# Определим сеть, повторяющую архитектуру из scratch-версии:
class TorchNet(nn.Module):
    def __init__(self):
        super(TorchNet, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TorchNet().to(device)

# Оптимизатор и критерий
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Обучение
n_epochs = 10
for epoch in range(1, n_epochs+1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}/{n_epochs}, Loss: {avg_loss:.4f}")

    # Валидация
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
    val_acc = correct / len(val_loader.dataset)
    print(f"  Val accuracy: {val_acc:.4f}")

# Финальная проверка на тесте
model.eval()
correct = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
test_acc = correct / len(test_loader.dataset)
print(f"Test accuracy: {test_acc:.4f}")

