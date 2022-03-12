import string

class Test:
    def assert_equals(self, function, answer):
        if function == answer:
            print('Nice')
        else:
            print('no good')


def is_wristband(lst):
    
    horizontal = list(dict.fromkeys([len(list(dict.fromkeys(i))) for i in lst]))#horizontal
    vertical = list(dict.fromkeys([len(list(dict.fromkeys(i))) for i in list(zip(*lst))])) #vertical
    leftright = list(dict.fromkeys([len(list(dict.fromkeys(i))) for i in [[c for c in r if c is not None] for r in zip(*[([None] * (len(lst) - 1))[i:] + r + ([None] * (len(lst) - 1))[:i] for i, r in enumerate(lst)])]]))
    rightleft = list(dict.fromkeys([len(list(dict.fromkeys(i))) for i in [[c for c in r if c is not None] for r in zip(*[([None] * (len(lst) - 1))[:i] + r + ([None] * (len(lst) - 1))[i:] for i, r in enumerate(lst)])]]))

    return True if [1] in [horizontal, vertical, leftright, rightleft] else False


Test = Test()

Test.assert_equals(is_wristband( 
[['A', 'A'], 
['B', 'B'], 
['C', 'C']]), True)

Test.assert_equals(is_wristband(
[['A', 'B'], 
['A', 'B'], 
['A', 'B']]), True)

Test.assert_equals(is_wristband(
[['A', 'B', 'C'], 
['C', 'A', 'B'], 
['B', 'C', 'A'], 
['A', 'B', 'C']]), True)

Test.assert_equals(is_wristband(
[['A', 'B', 'C'], 
['C', 'A', 'B'], 
['D', 'C', 'A'], 
['E', 'D', 'C']]), True)

Test.assert_equals(is_wristband(
[['A', 'B', 'C'], 
['B', 'A', 'B'], 
['D', 'C', 'A'], 
['E', 'D', 'C']]), False)

Test.assert_equals(is_wristband(
[['A', 'B', 'C'], 
['B', 'C', 'A'], 
['C', 'A', 'B'], 
['A', 'B', 'A']]), True)

Test.assert_equals(is_wristband(
[['A', 'B', 'C'], 
['B', 'C', 'D'], 
['C', 'D', 'E'], 
['D', 'E', 'F']]), True)

Test.assert_equals(is_wristband(
[['A', 'B', 'C'], 
['B', 'C', 'D'], 
['C', 'D', 'E'], 
['D', 'E', 'E']]), True)

Test.assert_equals(is_wristband(
[['A', 'B', 'C'], 
['B', 'C', 'D'], 
['C', 'D', 'E'], 
['D', 'F', 'E']]), False)

Test.assert_equals(is_wristband(
[['A', 'B', 'C'], 
['B', 'D', 'A'], 
['C', 'A', 'B'], 
['A', 'B', 'A']]), False)

Test.assert_equals(is_wristband(
[['A', 'B'],  
['A', 'B'], 
['A', 'C'],
['A', 'B']]), False)

Test.assert_equals(is_wristband(
[['A', 'A'],
['B', 'B'],
['C', 'C'],
['D', 'B']]), False)
 
Test.assert_equals(is_wristband(
[['A', 'A'],
['B', 'B'],
['C', 'C'],
['C', 'C']]), True)
