def left_slide(row):
    count0 = row.count(0)
    for i in range(count0):
        row.remove(0)
        row.append(0)
    for i in range(len(row)-1):
        if row[i] == row[i+1]:
            row[i] += row[i+1]
            row[i+1] = 0
    count0 = row.count(0)
    for i in range(count0):
        row.remove(0)
        row.append(0)
    return row

    
print(left_slide([2, 2, 2, 0]))
print(left_slide([2, 2, 4, 4, 8, 8]))
print(left_slide([0, 2, 0, 2, 4]))
print(left_slide([0, 2, 2, 8, 8, 8]))
print(left_slide([0, 0, 0, 0]))
print(left_slide([0, 0, 0, 2]))
print(left_slide([2, 0, 0, 0]))
print(left_slide([8, 2, 2, 4]))
print(left_slide([1024, 1024, 1024, 512, 512, 256, 256, 128, 128, 64, 32, 32]))
