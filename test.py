from fann2 import libfann
from terminaltables import AsciiTable
import random

ann = libfann.neural_net()
ann.create_from_file('semeion.net')

samples = []

with open('semeion.data') as semeion_file:
    for line in semeion_file:
        numbers = line.split(' ')
        samples.append(map(lambda x: float(x), numbers[0:256]))


# computes result from [1x256] sample, requires first_layer and second_layer to be defined globally
# returns single detected number
def compute_result(input_sample):
    # process input vector through both layers on NN
    result = ann.run(input_sample)

    # loop through all numbers in sequence and return index of highest value
    maximum = 0
    selected_index = 0
    for index in xrange(10):
        if result[index] > maximum:
            maximum = result[index]
            selected_index = index

    return selected_index


# converts [1x256] sample line into pretty 16x16 character block where 1 is * and other symbols are omitted
def print_sample(input_sample):
    text = []

    # process sample row by row
    for sample_row in xrange(16):
        text_row = input_sample[sample_row*16:(sample_row + 1)*16]
        # replace 1 with * and 0 with empty space
        text_row = map(lambda cell: '*' if cell == 1 else ' ', text_row)
        # join 16 characters into line
        text_row = ''.join(text_row)
        # line to rows array
        text.append(text_row)

    # finally, join rows with newlines
    return '\n'.join(text)

print 'Actual testing of trained NN'

table_data = [
    ['Sample', 'Digit', 'Sample', 'Digit', 'Sample', 'Digit', 'Sample', 'Digit']
]

# we print three rows
for row in xrange(3):
    table_data.append([''] * 8)
    # with 8 columns, 4 image -> result pairs
    for col in xrange(4):
        # pick one random sample between 0 and sample count
        ri = random.randint(0, len(samples) - 1)
        sample = samples[ri]

        table_data[row+1][col*2] = print_sample(sample)
        table_data[row+1][col*2+1] = '\n'.join([' ' * 5, ' ' * 5, '  %d' % compute_result(sample)])

table = AsciiTable(table_data)
table.inner_row_border = True

print table.table