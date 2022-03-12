def shift_letters(txt, n):
    temptxt = [i for i in txt if i != ' ']
    [temptxt.insert(0, temptxt.pop()) for _ in range(n)]
    [temptxt.insert(i, ' ') for i in [i[0] for i in enumerate(txt) if i[1] == ' ']]
    return ''.join(temptxt)

shift_letters("Made by Harith Shah", 15)
shift_letters("Boom", 1)
shift_letters("The most addictive way to learn", 19)
shift_letters("This is a test", 13)
shift_letters("Shift the letters", 1)
shift_letters("A B C D E F G H", 4)
shift_letters("Edabit helps you learn in bitesize chunks", 39)
shift_letters("To be or not to be", 6)
shift_letters("Made by Harith Shah", 18)
shift_letters("Boom", 0)
shift_letters("The most addictive way to learn", 5)
shift_letters("This is a test", 9)
shift_letters("Shift the letters", 3)
shift_letters("A B C D E F G H", 10)
shift_letters("Birds of a Feather Flock Together", 32)
