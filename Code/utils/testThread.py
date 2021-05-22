import threading
import requests
from lxml import etree
from urllib import request
from queue import Queue
from bs4 import BeautifulSoup
from requests.api import post

class Producer(threading.Thread):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36'
    }
    def __init__(self,page_queue,result_queue,*args,**kwargs):
        super(Producer, self).__init__(*args,**kwargs)
        self.page_queue = page_queue
        self.result_queue = result_queue
 
    def run(self):
        while True:
            if self.page_queue.empty():
                print('bye')
                break
            print('剩余页数：', page_queue.qsize())
            url = self.page_queue.get()
            self.parse_page(url)
 
    def parse_page(self,u):
        try:
            html = requests.get(u)
            html.encoding = 'utf8'
            h = html.text
            bs = BeautifulSoup(h)
            lis = bs.find('ul',{'id':'list'}).findAll('li')
            dic = {}
            for i in lis:
                self.result_queue.put((i.span.text.strip('[]'),i.a.text))
                dic[i.span.text.strip('[]')] = i.a.text
            print(u, dic)
        except Exception as e:
            pass


N = 45836
N_threads = 100
page_queue = Queue(N)
result_queue = Queue(N)
url = 'http://www.ztflh.com/?c='

for i in range(N):
    u = url + str(i)
    page_queue.put(u)

for x in range(N_threads):
    t = Producer(page_queue, result_queue)
    t.start()

result = list(result_queue.queue)
result.sort()

with open('中图分类号.csv', 'w', encoding='utf_8_sig') as file:
    for k,v in result:
        file.write(k.strip('{}')+','+v+'\n')
