import numpy as np
import pickle
from mlxtend.data import loadlocal_mnist
import math as m
from scipy import signal

np.random.seed(1)                   # заставим numpy выдавать одинаковые набор случайных чисел для каждого запуска программы
np.set_printoptions(suppress=True)  # выводить числа в формате 0.123 а не 1.23e-1

# В `X` находятся изображения для обучения, а в `y` значения соответственно
# `X.shape` == (60000, 784)   # изображения имеют размер 28x28 pix => 28*28=784
# `y.shape` == (60000,)       # каждое значение это число от 0 до 9 то что изображено на соответствующем изображении 
X, y = loadlocal_mnist(
        images_path="/home/andrey/datasets/mnist/train-images-idx3-ubyte", 
        labels_path="/home/andrey/datasets/mnist/train-labels-idx1-ubyte")

# В `Xt` находятся изображения для тестирования, а в `yt` значения соответственно
# `Xt.shape` == (10000, 784)   # изображения имеют размер 28x28 pix => 28*28=784
# `yt.shape` == (10000,)       # каждое значение это число от 0 до 9 то что изображено на соответствующем изображении 
Xt, yt = loadlocal_mnist(
        images_path="/home/andrey/datasets/mnist/t10k-images-idx3-ubyte", 
        labels_path="/home/andrey/datasets/mnist/t10k-labels-idx1-ubyte")

def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)

    return 1.0 / (1.0 + np.exp(-x))

def convert(y):
    y_d = np.zeros((len(y), 10))

    for idx, val in enumerate(y):
        y_d[idx, val] = 1.0

    return y_d

X = X * (1 / 255)
Xt = Xt * (1 / 255)

# Параметры:

lr = 1       # значени на которое будет домножаться дельта на каждом шаге
batch = 60   # кол-во изображений использованное для обучения на каждом шаге
epochs = 100  # кол-во эпох. Если видно что прогресс есть, но нужно больше итераций 

class MnistModel:
    def __init__(self, lr=0.1, batch=60):
        self.lr = lr
        self.batch = batch

        self.W_conv = np.random.uniform(-0.05, 0.05, (8, 3, 3))
        self.W_linear = np.random.uniform(-0.05, 0.05, (8 * 26 * 26, 10))

    def load(self, conv, linear):
        with open(conv, 'rb') as f:
            self.W_conv = np.array(pickle.load(f)).reshape((8, 3, 3))

        with open(linear, 'rb') as f:
            self.W_linear = np.array(pickle.load(f)).reshape((8 * 26 * 26, -1))

    def linear_forward(self, X):
        return np.dot(X, self.W_linear)

    def sigmoid_forward(self, X):
        return sigmoid(X)

    def relu_forward(self, X):
        X_o = X.copy()
        X_o[X < 0] = 0

        return X_o
    
    def convolution_forward(self, X):
        def img2col(X):
            (batch_size, ch, rows, cols) = X.shape
            (out_rows, out_cols) = (rows - 2, cols - 2)
            wcount = out_rows * out_cols 

            res = np.empty((batch_size * wcount, ch * 3 * 3))

            for batch in range(0, batch_size):
                for orow in range(0, out_rows):
                    for ocol in range(0, out_cols):
                        row = orow
                        col = ocol
                        res[batch * wcount + orow * out_cols + ocol] = X[batch][:, row: row + 3, col: col + 3].flatten()
            
            return res

        def col2img(X, bs, rows, cols):
            (wcount, filters) = X.shape

            batch_size = rows * cols

            res = np.empty((bs, filters, rows, cols))

            for i in range(0, wcount):
                batch = i // batch_size
                offset = (i % batch_size)
                row = offset // cols
                col = offset % cols

                res[batch, :, row, col] = X[i]

            return res

        def gemm_convolution(X, W):
            cols = img2col(X)
            res = cols.dot(np.tile(W.reshape((8, -1)).T, X.shape[1]))

            return col2img(res, len(X), 26, 26)

        def convolve(X, W):
            (in_rows, in_cols) = X.shape
            (f_row, f_cols) = W.shape
            (out_rows, out_cols) = (in_rows - 2, in_cols - 2)

            out = np.empty((out_rows, out_cols))

            for orow in range(0, out_rows):
                for ocol in range(0, out_cols):
                    row = orow
                    col = ocol

                    out[orow][ocol] = np.sum(X[row: row + f_row, col: col + f_cols] * W)

            return out


        def convolution1(X, W):
            (filter_channels, filter_height, filter_width) = W.shape
            (batch_size, in_channels, in_rows, in_cols) = X.shape
            (out_channels, out_rows, out_cols) = (filter_channels, in_rows - 2, in_cols - 2)

            res = np.zeros((batch_size, out_channels, out_rows, out_cols))

            for batch in range(0, batch_size):
                for och in range(0, out_channels):
                    for ich in range(0, in_channels):
                        res[batch][och] += signal.correlate2d(X[batch][ich], W[och], mode='valid')

            return res


        def convolution(X, W):
            (filter_channels, filter_height, filter_width) = W.shape
            (batch_size, in_channels, in_rows, in_cols) = X.shape
            (out_channels, out_rows, out_cols) = (filter_channels, in_rows - 2, in_cols - 2)

            res = np.empty((batch_size, out_channels, out_rows, out_cols))

            for batch in range(0, batch_size):
                print(batch)

                for och in range(0, out_channels):
                    for orow in range(0, out_rows):
                        for ocol in range(0, out_cols):
                            out = 0

                            row = orow
                            col = ocol

                            for ch in range(0, in_channels):
                                # for h in range(0, filter_height):
                                #     for w in range(0, filter_width):
                                #         out += X[batch][ch][row + h][col + w] * W[och][h][w]
                                
                                # немного ускорим процесс
                                out += np.sum(X[batch][ch][row: row + filter_height, col: col + filter_width] * W[och])
                            
                            res[batch][och][orow][ocol] = out
            return res
    
        return convolution1(X, self.W_conv)

    def forward(self, X):
        conv_1 = self.convolution_forward(X)
        relu_1 = self.relu_forward(conv_1)
        flatten_1 = relu_1.reshape(len(X), -1)
        linear_1 = self.linear_forward(flatten_1)
        sigmoid_1 = self.sigmoid_forward(linear_1)
        
        return  sigmoid_1


if __name__ == "__main__":
    model = MnistModel()
    model.load("W_conv.pickle", "W_linear.pickle")

    # Xt = Xt[0:1000]
    # yt = yt[0:1000]

    tp = model.forward(Xt.reshape((-1, 1, 28, 28)))

    print((np.sum(yt == np.argmax(tp, axis=1)) / len(yt))) 
