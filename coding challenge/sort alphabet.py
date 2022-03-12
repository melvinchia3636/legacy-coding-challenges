import string

def sorting(s):
    return ''.join([i[1] for i in sorted([(([sub[item] for item in range(len(string.ascii_lowercase)) for sub in [string.ascii_lowercase, string.ascii_uppercase]] + list(string.digits)).index(i), i) for i in s], key = lambda k: k[0])])
