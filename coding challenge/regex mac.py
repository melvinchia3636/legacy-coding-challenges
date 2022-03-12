import re

txt1 = '01:32:54:67:89:AB'

pattern = '((?:[0-9]|[A-F])(?:[0-9]|[A-F])(?::)){5}(?:[0-9]|[A-F])(?:[0-9]|[A-F])'

print(bool(re.match(pattern, txt1)))
