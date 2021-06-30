# Softmax Formula
import numpy as np

#inputs = np.array([1, 4, 9, 7, 5])
inputs = np.array([3, 1, -3])

def softmax(inputs):
    return np.exp(inputs)/sum(np.exp(inputs))

outputs = softmax(inputs)
print(outputs)

for n in range(len(outputs)):
    print('{} -> {}'.format(inputs[n], outputs[n]))

print(sum(outputs)) # 1.0