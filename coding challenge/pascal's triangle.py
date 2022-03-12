def pascals_triangle(row):
    triangle = [[1, 1]]
    triangle = [triangle := [1]+[triangle[i]+triangle[i+1] for i in range(len(triangle)-1)]+[1] for i in range(row)][-1]
    return ' '.join([str(i) for i in triangle])

	
print(pascals_triangle(5))


