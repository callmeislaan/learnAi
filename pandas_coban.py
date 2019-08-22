import pandas as pd
import numpy as np

# pandas series

# a = pd.Series(data = [1, 2, 3, 4],
#             index = ['a', 'b', 'c', 'd'])

# print(a)

# print(a['a'])
# print(a[0])

# # thay doi
# a['b'] = 20
# print(a)

# print(a**2)
# print(a - 2)
# print(a + 2)
# print(a*2)
# print(a/2)

# print(np.power(a, 2))

data = [{'a' : 1, 'b' : 2, 'c' : 3, 'y' : 4, 'z' : 5}, 
        {'x' : 2, 'y' : 3, 'z' : 4, 'a' : 1}]
    
df = pd.DataFrame(data, index = ['i1', 'i2'])

print(df)
print('size: ', df.size)
print('shape: ', df.shape)
print('index: ', df.index)
print('columns: ', df.columns)
print()
print(df.loc[['i1']])
print(df.loc[['i2']])
print(df.iloc[[0]])
print(df.iloc[[1]])
print(df[['a', 'b']])