# hand_write_digit_recognition_NN
using neural network to recognize hand write digit

I am using one input layer which has 785 nodes, one hidden layer which has 101 nodes and an output layer which has 10 nodes. Input layer is completely connected with hidden layer and also hidden layer is completely connected with output layer. The input data is pixel values from a 28*28 image. The label data is the number appears on the image.

Input layer recieves data from image and there's a bias node has a constant value 1. Input layer processes the data and pases to hidden layer. Hidden layer recieves the processed values from input layer and there's also a bias node has a constant value 1. Then, after hidden layer processes the data and passes to out layer, the number with highest values is the predict number. I examine the model with test set which has 10000 image and print out the accuracy.

After 60000 images training, the model performs a 95% accuracy on 10000 images test set.

The learning rate is set to 0.1 and momentum is set to 0.9.


 
