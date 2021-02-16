import pandas as pd

x=[1,2,3,4,5]
x =pd.Series(x)

x = pd.Series(x, index=['a','b','c','d','e'])
print(x['a'])
print(x[0])
print(x['a','e'])
print(x.a)

y = {"수학": 90, "영어":80 ,"과학": 95,"미술":80}
y = pd.Series(y)
print(y)
print(y['수학'])
print(y['영어':]) # :뒤 값인 맨 마지막까지 포함한다