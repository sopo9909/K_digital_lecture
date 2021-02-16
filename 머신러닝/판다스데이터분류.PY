import pandas as pd

title=pd.read_csv('titles.csv')
print(len(title))
print(title.count())
print(title.sort_values(by='year',ascending=True).head(2))
Hamlet=title['title']=='Hamlet'
print(title[Hamlet])
Treasureisland=title['title']=='Treasure Island'
print(title[Treasureisland].sort_values(by='year',ascending=True))
up1950=title['year']>=1950
down1959=title['year']<=1959
new=title[up1950&down1959]
print(len(new))