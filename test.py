from itertools import zip_longest
import timeit

def has_consecutive_series(l, m): l = [sum(i) for i in zip_longest(l, m, fillvalue=0)]; return sum(range(min(l), max(l)+1)) == sum(l)

start = timeit.default_timer()

print(has_consecutive_series([1, 2, 3], [1, 1, 1]))
print(has_consecutive_series([1, 2, 3], [1, 2, 1]))
print(has_consecutive_series([4, 6, -5, 8, 4], [-2, -3, 9, -3, 2]))
print(has_consecutive_series([12, 3], [0, 10, 14, 15, 16]))
print(has_consecutive_series([8, 6, 10], [25, 28, 25, 26, 28, 29]))
print(has_consecutive_series([11, 5, 3], [-2, 5, 8, 12]))
print(has_consecutive_series([11, 5, 3], [-2, 5, 8, 11]))

stop = timeit.default_timer()

print(round(stop - start, 6))