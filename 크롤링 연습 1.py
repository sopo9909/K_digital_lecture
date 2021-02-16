import requests
from bs4 import BeautifulSoup

URL='https://scrapying-study.firebaseapp.com/01/'
response=requests.get(URL)
soup=BeautifulSoup(response.text,"html.parser")
result=soup.find("table")
result2=result.find_all("th",limit=2)
result3=result.find_all("td")
a=[]
dicdic={}
ax=[]
for i in result3:
    ax.append(i.text)
for x in range(0,len(ax)-1,2):
    dicdic = dict(이름=ax[x],나이=ax[x+1])
    a.append(dicdic)
print(a)