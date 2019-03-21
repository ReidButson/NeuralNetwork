import numpy as np

def synapse(inputs, weights, threshold):
    x = np.dot(np.array(inputs), np.array(weights)) - np.matrix(threshold)
    return np.array(1 / (1 + np.exp(-x)))

def gradient_output(desired, actual):
    return np.array(actual * (1 - actual) * (desired - actual))


def gradient_hidden(_synapse, gradient, weights):
    x = np.array(_synapse * (1 - _synapse))
    y = np.array(np.dot(weights, gradient.transpose()).transpose())
    return np.array(x*y)

def weight_delta(learn_rate, prev_synapse, gradient):
    x = learn_rate * (np.matrix(prev_synapse).transpose() * gradient)
    return x

def threshold_delta(learn_rate, gradient):
    return -learn_rate * gradient

if __name__ == '__main__':

    in1 = np.matrix([[1, 1]])
    wm1 = np.matrix([[0.5, 0.9], [0.4, 1.0]])
    th1 = np.matrix([[0.8, -0.1]])
    
    s1 = synapse(in1, wm1, th1)

    wm2 = np.matrix([[-1.2], [1.1]])
    th2 = np.matrix([[0.3]])
    
    s2 = synapse(s1, wm2, th2)

    des = np.matrix([[0]])

    df = gradient_output(des, s2)
    hf = gradient_hidden(s1, df, wm2)
    
    # learning rate
    a = 0.1
    
    wd2 = weight_delta(a, s1, df)
    td2 = threshold_delta(a, df)
    
    wd1 = weight_delta(a, in1, hf)
    td1 = threshold_delta(a, hf)
    
    
    print(s1)
    print('-'*50)
    print(s2)
    print('='*50)
    print(df)
    print('-'*50)
    print(hf)
    print('-'*50)
    print(wd1)
    print('-'*50)
    print(wd2)
    print('-'*50)
    print(td1)
    print('-'*50)
    print(td2)