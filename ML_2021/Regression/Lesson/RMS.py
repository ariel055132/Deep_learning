# Root mean square implememtation
import numpy as np

def rmse(prediction, target):
    difference = np.subtract(prediction, target)
    squared_difference = difference ** 2
    mean_of_squared_difference = squared_difference.mean()
    rmse_val = np.sqrt(mean_of_squared_difference)
    return rmse_val

y_actual = [1,2,3,4,5]
y_predicted = [1.6,2.5,2.9,3,4.1]

result = rmse(y_predicted, y_actual)
print(result) # 0.6971370023173351