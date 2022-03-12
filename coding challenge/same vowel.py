def same_vowel_group(w):
    v = []
    for i in w[0]:
        for j in 'aeiou':
            if j in i:
                v.append(j)
    lol = []
    [lol.append(i) for i in v if i not in lol]
    not_contain = [i for i in 'aeiou' if i not in lol]
    [w.remove(i) for i in w]
                
    print(lol)

same_vowel_group(["hoops", "chuff", "bot", "bottom"])
