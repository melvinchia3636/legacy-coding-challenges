def clean_up(files, sort):
    if sort == 'prefix':
        files = [i.split('.') for i in files]
        filename = [i[0] for i in files]
        uniquefilename = []
        final = []
        for i in filename:
            if i not in uniquefilename:
                uniquefilename.append(i)
        print(uniquefilename)
        for i in range(len(uniquefilename)):
            final.append([])
            for j in files:
                if uniquefilename[i] in j:
                    final[i].append('.'.join(j))
        return final
            
    if sort == 'suffix':
        files = [i.split('.') for i in files]
        filename = [i[1] for i in files]
        uniqueext = []
        final = []
        for i in filename:
            if i not in uniqueext:
                uniqueext.append(i)
        for i in range(len(uniqueext)):
            final.append([])
            for j in files:
                if uniqueext[i] in j:
                    final[i].append('.'.join(j))
        return final
    
print(clean_up(["ex1.html", "ex1.js", "ex2.html", "ex2.js"], "suffix"))
