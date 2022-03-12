def overlap(s1, s2):
    for i in range(len(s1)):
        if s2.startswith(s1[i:]):
            return s1[:i]+s2
    return s1+s2

print(overlap("sweden", "denmark"))
print(overlap("edabit", "iterate"))
print(overlap("honey", "milk"))
print(overlap("dodge", "dodge"))
print(overlap("colossal", "alligator"))
print(overlap("leave", "eavesdrop"))
print(overlap("joshua", "osha"))
print(overlap("diction", "dictionary"))
print(overlap("massive", "mass"))
