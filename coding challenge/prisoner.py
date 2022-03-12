def freed_prisoners(prison):
    count = 0
    if prison[0] != 0:
        for i in range(len(prison)):
            if prison[i] == 1 and prison[0] != 0:
                count += 1
                for j in range(len(prison)):
                    prison[j] = 0 if prison[j] == 1 else 0
        return count

print(freed_prisoners([1, 1, 0, 0, 0, 1, 0]))
print(type(freed_prisoners))
