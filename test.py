import numpy as np

arr = np.array([10, 20, 30, 40, 50])

# Take a slice (this is a VIEW)
my_slice = arr[1:4]  # [20, 30, 40]

# Modify the slice
my_slice[0] = 999

print(my_slice)  # [999,  30,  40]
print(arr)  # [ 10, 999,  30,  40,  50]  <-- Original mutated!