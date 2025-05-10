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
