# encoding: utf-8
import numpy as np
import struct
import random
import math
import csv


hidden_unit = 100         # numbers of hidden units
momentum_value = 0.9     # value of momentum
learning_rate = 0.1      # learning rate


def initial_w(n1, n2):
    """
    initialize w with n1 unit number and n2 output numbers, the w range is from 0.05 to -0.05
    """
    return np.random.uniform(-0.05, 0.05, (n1,n2)).astype(np.float64)

def sigmoid(x, derivative=False):
    """
    sigmoid function
    """
    return x*(1-x) if derivative else 1/(1+np.exp(-x))

def matrix_multification(first, second):
    """
    matrix multiply function
    """
    return sigmoid(first.dot(second))

def predict_digit(outcome):
    """
    convert output layer values to predict target
    """
    big = np.max(outcome)
    for i in range(10):
        if outcome[i] == big:
            return i
def accuracy(test_images, test_labels, w1, w2):
    """
    calculate the accuarcy by using 10000 test images
    """
    n = len(test_images)
    count = 0
    for i in range(n):
        temp = np.insert(test_images[i], 0, 1, 0)
        hidden_values = matrix_multification(temp, w1)
        temp = np.insert(hidden_values, 0, 1, 0)
        y_values = matrix_multification(temp, w2)
        if predict_digit(y_values) == test_labels[i]:
            count += 1
    return count*100/n

def create_task_values(label):
    """
    generate task values by using taining label
    """
    temp = np.empty(10)
    temp.fill(0.1)
    temp[int(label)] = 0.9
    return temp


def train(w1_t, w2_t, images, labels, rate, momentum, t_images, t_labels):
    """
    train the model, update w1 and w2 and check the accuracy in certain period
    """
    w1_va = np.zeros_like(w1_t)
    w2_va = np.zeros_like(w2_t)
    n = len(images)
    for i in range(n):
        if i%1000==0 :
            print("After %d images training is" % i)
            print(accuracy(t_images, t_labels, w1_t, w2_t), "%")
        w1_t, w2_t, w1_va, w2_va = upgrade(w1_t, w2_t, images[i], labels[i], rate, momentum, w1_va, w2_va)
    print("Final accuracy is: ")
    print(accuracy(t_images, t_labels, w1_t, w2_t))

def upgrade(w1_u, w2_u, image, label, rate, momentum, w1_va, w2_va):
    """
    update w1, w2, w1 variation and w2 variation 
    """
    temp = np.insert(image, 0, 1, 0)
    hidden_values = matrix_multification(temp, w1_u)
    temp = np.insert(hidden_values, 0, 1, 0)
    y_values = matrix_multification(temp, w2_u)
    task_values = create_task_values(label)

    w2_error = y_values * (1-y_values) * (task_values - y_values)
    w1_error = (w2_error.dot(w2_u.T))[1::] * hidden_values * (1 - hidden_values)
    w2_adjust = np.ones_like(w2_u) * rate * ([w2_error, ]*len(w2_u))
    for i in range(1, len(w2_adjust)):
        w2_adjust[i] = w2_adjust[i] * hidden_values[i-1]
    w2_adjust = w2_adjust + momentum * w2_va
    w2_u = w2_u + w2_adjust
    w2_va = w2_adjust

    w1_adjust = np.ones_like(w1_u) * rate * ([w1_error, ]*len(w1_u))
    for i in range(1, len(w1_adjust)):
        w1_adjust[i] = w1_adjust[i] * image[i-1]
    w1_adjust = w1_adjust + momentum * w1_va
    w1_u = w1_u + w1_adjust
    w1_va = w1_adjust

    return w1_u, w2_u, w1_va, w2_va

def run():
    """
    running function 
    """
    with open('mnist_train.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))
    train_images = np.array(data).astype(np.float64)
    train_labels = train_images[:,0]
    train_images = np.delete(train_images, 0, 1)
    train_images = train_images/255
    #normalize the pixel values

    with open('mnist_test.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))
    test_images = np.array(data).astype(np.float64)
    test_labels = test_images[:,0]
    test_images = np.delete(test_images, 0, 1)
    test_images = test_images/255
    #normalize the pixel values

    number = len(train_images)
    w1 = initial_w(785, hidden_unit)
    w2 = initial_w(hidden_unit+1, 10)
    
    train(w1, w2, train_images, train_labels, learning_rate, momentum_value, test_images, test_labels)

    
if __name__ == '__main__':
    run()