#desigend by Hoyoung
#요곳이 재미짐
import csv
from bs4 import BeautifulSoup
import requests
import pandas as pd
call_rate=[]
call_date=[]
def print_stock_price(page_num):
    result = [[], []]
    for n in range(page_num):
        url = 'https://finance.naver.com/marketindex/interestDailyQuote.nhn?marketindexCd=IRR_CALL&page='+str(n+1)
        r = requests.get(url)
        html = r.content
        soup = BeautifulSoup(html, 'html.parser')
        tr = soup.select('table > tbody > tr')
        for i in range(1, len(tr)-1):
            if tr[i].select('td')[0].text.strip():
                a=tr[i].select('td')[0].text.strip()
                b=tr[i].select('td')[1].text.strip()
                call_rate.append(b)
                call_date.append(a)
    call_data={'date':call_date,'콜금리':call_rate}
    call_data=pd.DataFrame(call_data)
    call_data.to_csv('call_rate1.csv',encoding='utf-8')
print_stock_price(573)
print("끝")
#---기준금리
date=[]
nom_rate=[]
url1 = 'https://www.bok.or.kr/portal/singl/baseRate/list.do?dataSeCd=01&menuNo=200643'
r = requests.get(url1)
html = r.content
soup = BeautifulSoup(html, 'html.parser')
tr = soup.select('.fixed tbody tr')
for i in range(1, len(tr)-1):
    if tr[i].select('td')[0].text.strip():
        a=tr[i].select('td')[0].text.strip()+'.'
        b=tr[i].select('td')[1].text.strip().replace('월 ','.').replace('일','')
        c=tr[i].select('td')[2].text.strip()
        date.append(a+b)
        nom_rate.append(c)
nom_rate={'날짜':date,'기준금리':nom_rate}
nom_rate=pd.DataFrame(nom_rate)
nom_rate.to_csv('nom_rate.csv',encoding='utf-8')
print("끝")
# 뉴스 url =https://search.naver.com/search.naver?&where=news&query=%EA%B8%88%EB%A6%AC&sm=tab_pge&sort=0&photo=0&field=0&reporter_article=&pd=0&ds=&de=&docid=&nso=so:r,p:all,a:all&mynews=1&cluster_rank=52&start=1&refresh_start=0