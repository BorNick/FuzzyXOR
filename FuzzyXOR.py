import NeuroNet as net

nn = net.NeuroNet("Parameters.txt")

# nn = net.NeuroNet()

nn.train("TrainExamples.txt", 0.001, 100, 0.24)

# print(nn.w1)
