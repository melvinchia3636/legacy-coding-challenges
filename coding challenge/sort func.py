def advanced_sort(lst):
    ref = list(dict.fromkeys(lst))
    numlst = []
    final = []
    print(ref)
    for i in ref:
        numlst.append(lst.count(i))
    for i in range(len(ref)):
        final.append([ref[i] for i2 in range(numlst[i])])
    print(final)

advanced_sort([2,1,2,1])
