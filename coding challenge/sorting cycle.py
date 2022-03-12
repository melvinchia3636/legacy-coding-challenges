def cycle_length(lst, n):
    
    newlst = [sorted(lst).index(i)+1 for i in lst]
    n = newlst[lst.index(n)]
    lst = newlst
    
    current = n
    index = lst.index(current)
    run = True
    count = 0
    while n != index+1:
        current = lst[n-1]
        lst[lst.index(n)], lst[n-1] = lst[n-1], lst[lst.index(n)]
        n = current
        index = lst.index(n)
        count += 1
    print(count)
    return count
    
cycle_length([43, 81, 88, 93, 17, 32, 19, 11], 93)
