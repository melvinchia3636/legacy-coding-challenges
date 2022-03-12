def index_multiplier(lst):
    return sum(a*b for a, b in list(enumerate(lst)))

index_multiplier([-3, 0, 8, -6])
