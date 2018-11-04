import random

f = open("TrainExamples.txt", "w")
x = [0 for i in range(2)]
for i in range(1000):
    for j in range(2):
        x[j] = random.choice([0, 1])
    y = x[0] ^ x[1]
    for j in range(2):
        x[j] += random.normalvariate(0, 0.1)
    f.write(str(x[0]) + " " + str(x[1]) + " " + str(y) + "\n")
    # print(str(x[0]) + " ^ " + str(x[1]) + " = " + str(y))
f.close()
