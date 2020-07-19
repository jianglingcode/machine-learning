import  requests
from  lxml  import etree
import json
import time
import csv

url = 'https://search.jd.com/Search?'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36 Edg/83.0.478.56',
    'referer': 'https://shuma.jd.com/'
        }

param = {
    'keyword': '荣耀手机',
    'enc': 'utf-8',
    'pvid': 'a2e1d4e484c04e80af32664a8884c31e'
    }
res=requests.get(url, headers=headers,params=param)

selector = etree.HTML(res.text)
product_list = selector.xpath('//*[@id="J_goodsList"]/ul/li')

for product in product_list:
    p_id = product.xpath('@data-sku')[0]

    detail_url='https://item.jd.com/'+str(p_id)+'.html'
    res_detail=requests.get(detail_url,headers=headers)
    selector_detail = etree.HTML(res_detail.text)
    
    for  i   in  range(2):
        comment_url = 'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=' +\
            str(p_id) + '&score=0&sortType=5&page=' + str(i) +\
                '&pageSize=10&isShadowSku=0&fold=1'
        res_comment = requests.get(comment_url,headers=headers)
        time.sleep(3)
        #print(res_comment.encoding)
        res_comment_text = res_comment.text.replace('fetchJSON_comment98(','').replace(');','')
        comments = json.loads(res_comment_text)['comments']
        for  comment  in comments:
            commentdata = comment['content']
            element=[str(commentdata)]
            with open(r'text.csv','a',encoding='utf-8',newline='') as fp:
                writor=csv.writer(fp)
                writor.writerow(element) 

      