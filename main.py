import numpy as np
import network
from PIL import Image
import matplotlib.pyplot as plt


inputs = []
expected = []

for i in range(10):
    out = [0]*10
    out[i] = 1

    image = Image.open('./Images/Training/Train_{}.png'.format(i))
    length = image.size[0] * image.size[1]
    arr = np.asarray(image).reshape(1, length)[0]

    inputs.append(arr)
    expected.append(out)

binary_exp =[[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0]]

n = network.NeuralNet(45, 10, [5])

inp = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1]]
exp = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]

n.train(inputs, expected, 0.01)

plt.plot(n.sum_squares)
#plt.show()

print('1: ',np.around(n.activation(inputs[1])))
print('9: ',np.around(n.activation(inputs[9])))
print('2: ',np.around(n.activation(inputs[2])))
print('3: ',np.around(n.activation(inputs[3])))

print(inputs[1])
print(inputs[1].reshape(9, 5))
im = Image.fromarray(inputs[1].reshape(9, 5)*255, 'L')
im.thumbnail((45, 25))
im.save('./output.png')