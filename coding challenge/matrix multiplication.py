def matrix_mult(m1, m2):
    result = [[0, 0], [0, 0]]
    for i in range(len(m1)):
       for j in range(len(m2[0])):
           for k in range(len(m2)):
               result[i][j] += m1[i][k] * m2[k][j]
    return result

matrix_mult([[4, 2],[3, 1]], [[5, 6], [3, 8]])
