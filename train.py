from fann2 import libfann

# crating the NN object itself
ann = libfann.neural_net()

# We define a network that is:
# 1. fully connected
# 2. contains tree layers (input 256 neurons, then hidden of 256 neurons, and output of 10)
ann.create_standard_array([256, 256, 10])

# simple weight randomisation
ann.randomize_weights(-0.1, 0.1)

# learning rate means how much the delta will alter actual layer weights. 1 means simple addition, bet it can make
# you network "stuck", and slower learning will actually create better results
ann.set_learning_rate(0.7)

# sigmoid functions will be used as activation functions
ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC_STEPWISE)
ann.set_activation_function_output(libfann.SIGMOID_STEPWISE)

# we will load semeion.fann, process all samples at least 100 times (epochs), reporting after each one (1). The training
# will stop either after 100 epochs, or after the the average error will be less than 0.000001
ann.train_on_file('semeion.fann', 100, 1, 0.00001)

# we will save NN to semeion.net for later usage
ann.save('semeion.net')
