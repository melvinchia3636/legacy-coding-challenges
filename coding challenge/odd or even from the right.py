def char_at_pos(r, s):
    return list(reversed(list(reversed(r))[::2])) if s == 'odd' and type(r) is list else list(reversed(list(reversed(r))[1::2])) if s == 'even' and type(r) is list else ''.join(list(reversed(list(reversed(r))[::2]))) if s == 'odd' and type(r) is str else ''.join(list(reversed(list(reversed(r))[1::2]))) if s == 'even' and type(r) is str else 0
    
char_at_pos("EDABIT", "even")
char_at_pos("EDABIT", "odd")
char_at_pos("QWERTYUIOP", "even")
char_at_pos("POIUYTREWQ", "odd")
char_at_pos("ASDFGHJKLZ", "odd")
char_at_pos("ASDFGHJKLZ", "even")
char_at_pos([2, 4, 6, 8, 10], "even")
char_at_pos([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "odd")
char_at_pos(["!", "@", "#", "$", "%", "^", "&", "*", "(", ")"], "odd")
char_at_pos([")", "(", "*", "&", "^", "%", "$", "#", "@", "!"], "odd")
char_at_pos(["A", "R", "B", "I", "T", "R", "A", "R", "I", "L", "Y"], "odd")
