# these are lists tht will contain samples and results read from file
samples = []
results = []

# you need to put the semeion.data to the same directory from where you run the sample
with open('semeion.data') as semeion_file:
    for line in semeion_file:
        # line is being split into single strings by space
        numbers = line.split(' ')
        # this takes first 256 strings from list and append them to samples
        samples.append(numbers[0:256])
        # ...and the last 256 strings from list and append them to results
        results.append(numbers[256:266])

# created semeion.fann in the same directory
train = file('semeion.fann', 'w')
# basically writes "1593 256 10" as a first line of file. Note "\n" in the string - it adds newline
train.write('%d %d %d\n' % (len(samples), 256, 10))

# writes samples and results on separate lines
for index in xrange(len(samples)):
    train.write(' '.join(samples[index])+'\n')
    train.write(' '.join(results[index])+'\n')

train.close()
