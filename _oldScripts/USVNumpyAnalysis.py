import numpy

print("TRAIN NPY ARRAY:")
first_data_array = numpy.load('testOutputs/USVData/train.npy')
print("Array Summary:")
print(first_data_array)
print("Array Shape:")
print(first_data_array.shape)
print(numpy.count_nonzero(numpy.isnan(first_data_array)))