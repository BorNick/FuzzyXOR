import matplotlib.pyplot as pp
import random


class NeuroNet:

    def __init__(self, file_name = ""):
        # init parameters
        self.w1 = [[0 for i in range(3)] for j in range(2)]        # 1 layer weights + biases
        self.d_w1 = [[0 for i in range(3)] for j in range(2)]      # partial derivative of cf with respect to w1
        self.avg_d_w1 = [[0 for i in range(3)] for j in range(2)]  # average partial derivative of cf with respect to w1
        self.z = [0 for i in range(3)]                             # 1 layer output
        self.z[-1] = 1                                             # for bias
        self.d_z = [0 for i in range(3)]                           # partial derivative of cf with respect to z
        self.w2 = [0 for i in range(3)]                            # 2 layer weights + bias
        self.d_w2 = [0 for i in range(3)]                          # partial derivative of cf with respect to w2
        self.avg_d_w2 = [0 for i in range(3)]                      # average partial derivative of cf with respect to w2
        self.x = [0 for i in range(3)]                             # arguments
        self.x[-1] = 1                                             # for bias
        self.y = 0                                                 # result
        self.true_y = 0                                            # expected result
        self.d_cf = 0                                              # cost function derivative
        if file_name == "":
            self.gen_parameters()
        else:
            self.load_parameters(file_name)

    def write_matrix_to_file(self, mat, file):
        if type(mat) != list:
            file.write(str(mat) + "\n")
        elif type(mat[0]) != list:
            file.write(" ".join(str(elem) for elem in mat) + "\n")
        else:
            for line in mat:
                file.write(" ".join(str(elem) for elem in line) + "\n")

    def save_parameters(self, file_name):
        f = open(file_name, "w")
        f.write("W1\n")
        self.write_matrix_to_file(self.w1, f)
        f.write("W2\n")
        self.write_matrix_to_file(self.w2, f)
        f.close()

    def read_matrix_from_file(self, file):
        mat = []
        line = file.readline()[:-1].split(" ")
        is_float = True
        try:
            float(line[0])
        except ValueError:
            is_float = False
        while is_float:
            mat.append(list(map(float, line)))
            line = file.readline()[:-1].split(" ")
            try:
                float(line[0])
            except ValueError:
                is_float = False
        if len(mat) > 1:
            return mat
        elif len(mat[0]) > 1:
            return mat[0]
        else:
            return mat[0][0]

    def gen_parameters(self):
        amp = 5
        for i in range(2):
            for j in range(3):
                self.w1[i][j] = random.uniform(-amp, amp)
        for i in range(3):
            self.w2[i] = random.uniform(-amp, amp)

    def load_parameters(self, file_name):
        f = open(file_name, "r")
        f.readline()
        self.w1 = self.read_matrix_from_file(f)
        self.w2 = self.read_matrix_from_file(f)
        f.close()

    def layer1(self):
        for i in range(2):
            self.z[i] = 0
            for j in range(3):
                self.z[i] += self.x[j] * self.w1[i][j]
            self.z[i] = max(self.z[i], 0)

    def d_layer1(self):
        for i in range(2):
            for j in range(3):
                self.d_w1[i][j] = 0 if self.z[i] < 0 else self.d_z[i] * self.x[j]  # self.d_z[i] * self.x[j]
                self.avg_d_w1[i][j] += self.d_w1[i][j]

    def layer2(self):
        self.y = 0
        for i in range(3):
            self.y += self.z[i] * self.w2[i]

    def d_layer2(self):
        for i in range(3):
            self.d_w2[i] = self.d_cf * self.z[i]
            self.d_z[i] = self.d_cf * self.w2[i]
            self.avg_d_w2[i] += self.d_w2[i]

    def forward(self, x1, x2, y):
        self.x[0] = x1
        self.x[1] = x2
        self.layer1()
        self.layer2()
        cost = self.cost_func(self.y, y)
        result = 1 if self.y > 0.5 else 0
        return self.y, cost, result

    def func(self, x1, x2):
        self.x[0] = x1
        self.x[1] = x2
        self.layer1()
        self.layer2()
        if self.y > 0.5:
            return 1
        else:
            return 0

    def backward(self, result, expected_result):
        self.d_cost_func(result, expected_result)
        self.d_layer2()
        self.d_layer1()

    def cost_func(self, result, expected_result):
        return (result - expected_result) ** 2

    def d_cost_func(self, result, expected_result):
        self.d_cf = 2 * (result - expected_result)
        return self.d_cf

    def subtract_grad(self, learning_rate, batch_size):
        for i in range(2):
            for j in range(3):
                self.w1[i][j] -= learning_rate * (self.avg_d_w1[i][j] / batch_size)
                self.avg_d_w1[i][j] = 0
        for i in range(3):
            self.w2[i] -= learning_rate * (self.avg_d_w2[i] / batch_size)
            self.avg_d_w2[i] = 0

    def train(self, file_name, learning_rate, batch_size, avg_cost_dest):
        train_file = open(file_name, "r")
        lines = train_file.readlines()
        costs = []
        accuracies = []
        avg_cost = 1000
        while avg_cost >= avg_cost_dest:
            random.shuffle(lines)
            for i in range(int(len(lines) / batch_size)):
                if avg_cost < avg_cost_dest:
                    break
                avg_cost = 0
                accuracy = 0
                successes = 0
                attempts = 0
                for j in range(batch_size):
                    example = list(map(float, lines[i * batch_size + j][:-1].split(" ")))
                    result, cost, bin_result = self.forward(example[0], example[1], example[2])
                    avg_cost += cost
                    if bin_result == example[2]:
                        accuracy += 1
                        successes += 1
                    attempts += 1
                    self.backward(result, example[2])
                self.subtract_grad(learning_rate, batch_size)
                avg_cost /= batch_size
                accuracy /= batch_size
                costs.append(avg_cost)
                accuracies.append(successes / attempts)
                # print(avg_cost)
            # pp.figure(1)
            # pp.clf()
            # pp.plot(costs)
            # pp.pause(0.001)
            # pp.figure(2)
            # pp.clf()
            # pp.plot(accuracies)
            # pp.pause(0.001)
        # pp.plot(costs)
        pp.figure(1)
        pp.plot(costs)
        pp.figure(2)
        pp.plot(accuracies)
        pp.show()
        train_file.close()
        self.save_parameters("Parameters1.txt")
