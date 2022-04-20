from decimal import Decimal
from math import factorial

epsilon = 1e-7


def func(the_n):
    n = Decimal(the_n)
    fact = factorial(the_n)
    p = pow(fact, 1/n)
    return n / p


index = 1
result = None
while True:
    index += 100
    v1 = func(index)
    v2 = func(index + 1)
    delta = abs(v2 - v1)
    print(f'at {index+1}: value = {v2}, delta = {delta}')
    if delta < epsilon:
        result = v2
        break

print(f'Found at {index}: lim = {result}')