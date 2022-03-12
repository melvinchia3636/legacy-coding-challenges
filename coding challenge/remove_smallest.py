def remove_smallest(lst):
    lst.remove(min(lst)) if lst else lst; return lst

print(remove_smallest([1, 2, 3, 4, 5]))
