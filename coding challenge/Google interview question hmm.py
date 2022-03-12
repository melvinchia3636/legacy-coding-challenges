def add_one(lst):
    return [int(i) for i in list(str(int(''.join([str(i) for i in lst]))+1))]

import time

start = time.time()
print(add_one([1, 3, 2, 4]))
end = time.time()
print(end - start)

