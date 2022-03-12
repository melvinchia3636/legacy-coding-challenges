def get_lucky_number(size, nth):
    lst = list(range(1, size+1))[::2]
    count = 1
    while len(lst) > lst[count]:
        lst = [i for i in lst if (lst.index(i)+1) % lst[count] != 0]
        count += 1
    return lst[nth-1]
        
get_lucky_number(5000, 90)
