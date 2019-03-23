from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return np.array(1 / (1 + np.exp(-x)))


def clamp(x, low=0, high=1):
    return max(low, min(high, x))


def plot_weights(weights):
    high = max([len(x) for x in weights])
    for l, layer in enumerate(weights):
        a1 = len(layer)
        for i in range(a1):
            a2 = len(layer[0])

            for j in range(a2):
                weight = layer[i, j]
                r = sigmoid(weight)
                b = sigmoid(-weight)
                a = clamp(abs(weight))
                try:
                    plt.plot([l, l+1], [i*(high/a1) + (high/(2*a1)), j*(high/a2) + (high/(2*a2))], 'k-', lw=2, color=(r, 0, b, a), zorder=0)
                    if l == 0:
                        plt.scatter(l, i * (high / a1) + (high/(2*a1)), 80, color='black', marker='_', zorder=4)
                        plt.scatter(l, i * (high / a1) + (high/(2*a1)), 50, color='white', marker='_', zorder=5)
                    else:
                        plt.scatter(l, i * (high / a1) + (high/(2*a1)), 80, color='black', marker='o', zorder=4)
                        plt.scatter(l, i * (high / a1) + (high/(2*a1)), 50, color='white', marker='o', zorder=5)
                        if l == len(weights)-1:
                            plt.scatter(l + 1, j * (high / a2) + (high/(2*a2)), 80, color='black', zorder=5)
                            plt.scatter(l + 1, j * (high / a2) + (high/(2*a2)), 50, color='white', zorder=5)
                except ValueError:
                    print('Values are {}, 0, {}, {} weight is: {}'.format(r, b, a, weight))
    plt.show()


def scale(image, factor):
    size = tuple(np.array(image.size) * factor)
    return image.resize(size, Image.NEAREST)


def result_to_image(image, exp, value, raw):

    img = image.resize((50, 90), Image.NEAREST)

    img.convert('L')

    test_pixels = np.asmatrix(img) * 255

    # im.save('C:\\Users\\100561195\\Documents\\School\\MachineLearning\\MiniProject_3\\NeuralNetwork\\Images\\TestImage.png')

    canvas = Image.new('L', (450, 200), color=255)

    pixels = np.asmatrix(canvas)

    # Changes permissions from read only? some how?
    pixels = pixels + 0

    pixels[10:(10 + img.size[1]), 10:(10 + img.size[0])] = np.array(test_pixels)

    background = Image.fromarray(np.array(pixels))

    pencil = ImageDraw.Draw(background)

    font = ImageFont.truetype(font='times.ttf', size=20)

    pencil.text((100, 20), "Expected:", font=font, fill=0)
    pencil.text((100, 60), 'Actual:', font=font, fill=0)

    pencil.text((200, 20), str(exp), font=font, fill=0)
    pencil.text((200, 60), str(value), font=font, fill=0)

    for i, (val, raw_val) in enumerate(zip(list(range(len(raw))), raw)):
        pencil.text(((i * 40) + 10, 120), '+======', fill=0)
        pencil.text(((i * 40) + 10, 130), '| '+str(val), fill=0)
        pencil.text(((i * 40) + 10, 140), '|======', fill=0)
        pencil.text(((i * 40) + 10, 150), '| '+str(raw_val), fill=0)
        pencil.text(((i * 40) + 10, 160), '|______', fill=0)

    pencil.text(((10 * 40) + 10, 120), '|', fill=0)
    pencil.text(((10 * 40) + 10, 130), '|', fill=0)
    pencil.text(((10 * 40) + 10, 140), '|', fill=0)
    pencil.text(((10 * 40) + 10, 150), '|', fill=0)
    pencil.text(((10 * 40) + 10, 160), '|', fill=0)

    return background
