# coding: utf-8

import urllib2
import re
import sys
import base64
import os
import gevent
from gevent import monkey
monkey.patch_all()

root = 'taobaoDB'
category = ['cup','shoe','cloth','bike','keyboard']
rootURL = ['http://s.taobao.com/search?initiative_id=staobaoz_20141024&js=1&stats_click=search_radio_all%253A1&q=%B1%AD&tab=all&bcoffset=-4&s=',
        'http://s.taobao.com/search?initiative_id=staobaoz_20141024&js=1&stats_click=search_radio_all%253A1&q=%D0%AC&tab=all&bcoffset=-4&s=',
        'http://s.taobao.com/search?initiative_id=staobaoz_20141024&js=1&stats_click=search_radio_all%253A1&q=%D2%C2&tab=all&bcoffset=-4&s=',
        'http://s.taobao.com/search?initiative_id=staobaoz_20141024&js=1&stats_click=search_radio_all%253A1&q=%D7%D4%D0%D0%B3%B5&tab=all&bcoffset=-4&s=',
        'http://s.taobao.com/search?initiative_id=staobaoz_20141024&js=1&stats_click=search_radio_all%253A1&q=%BC%FC%C5%CC&tab=all&bcoffset=-4&s=']
        


def Crawler(rooturl):
    urls = set([])
    
    for page in range(0,1100,44):
        if len(urls) > 1000:
            break
            
        url = rooturl+str(page)
        for t in range(3):
            try:
                data = urllib2.urlopen(url).read()
                if len(data)>0:
                    break
            except Exception,e:
                print e
                
        if len(data)>0:
            links = re.findall('http://item.taobao.com[^"]+',data)
            links = links + re.findall('http://detail.tmall.com[^"]+',data)
            links = set(links)
            
            for link in links:
                urls.add(link)
        else:
            print 'Crawler:',rooturl,page,'skipped'
            
    print 'got',len(urls),'links in category',category[i]
    return urls
    
    
    
def GetPic(url, path):
    data = ''
    for i in range(3):
        try:
            data = urllib2.urlopen(url).read()
            if len(data) >= 0:
                break
        except Exception,e:
            print e
            print 'get error, try again'
            print 'url:',url
        

    title = data[data.find('<title>')+len('<title>'):data.find('</title>')]
    
    f = open(os.path.join(path,'info.txt'), 'w')
    f.write('title:\n')
    f.write(title)
    f.write('\nurl:\n')
    f.write(url.decode('gbk'))
    f.close()

    imgList = re.findall('"([^"]+.jpg)[^"]+.jpg', data)
    imgList = set(imgList)

    count = 1
    for img in imgList:
        get = ''
        name = ''
        if img.find('background') >= 0:
            get = re.findall('(http://[^"]+.jpg)',img)[0]
            name = os.path.join(path,'img_'+str(count)+'.jpg')
        else:
            get = re.findall('(http://[^"]+.jpg)',img)[0]
            name = os.path.join(path,'extra_'+str(count)+'.jpg')
                
        
        for t in range(3):
            imgdata = ''
            try:
                imgdata = urllib2.urlopen(get).read()
                if len(imgdata) > 0:
                    break
            except Exception,e:
                print e
                
        if len(imgdata) > 0:
            f = open(name, 'wb')
            f.write(imgdata)
            f.close
            count = count + 1
        else:
            print get,'skipped'
            continue
            
        print name,'done'
        
            

    

def Drive(pid):
    i = pid
    path = os.path.join(root,category[i])
    if os.path.exists(path) == False:
        os.mkdir(path)
       
    urlList = [] 
    listpath = os.path.join(root,category[i],'list')
    if os.path.exists(listpath) == False:
        urlList = Crawler(rootURL[i])
        f = open(listpath,'w')
        for url in urlList:
            f.write(url+'\n')
        f.close()
    else:
        f = open(listpath,'r')
        lines = f.readlines()
        for line in lines:
            if len(line)>0:
                urlList.append(line)


    index = 0
    breakp = os.path.join(root,category[i],'breakp')
    if os.path.exists(breakp):
        with open(breakp) as f:
            try:
                index = int(f.readline())
            except Exception,e:
                print e
        f.close()
            
    for url in urlList:
        path = os.path.join(root,category[i], str(index))
        if os.path.exists(path) == False:
            os.mkdir(path)
        GetPic(url,path)
        index = index + 1
        print index,'in',category[i],'done'
        
        f = open(breakp, 'w')
        f.write(str(index))
        f.close
    
    

if os.path.exists(root) == False:
    os.mkdir(root)
    
    

threads = [gevent.spawn(Drive, i) for i in range(len(category))]
gevent.joinall(threads)            





    
''''
[get]
tmall
<a href="#" style="background:url(http://gi4.md.alicdn.com/bao/uploaded/i4/653392567/T2.488X6XXXXXXXXXX_!!653392567.jpg_40x40q90.jpg) center no-repeat;">
taobao
<a href="#" style="background:url(http://img01.taobaocdn.com/bao/uploaded/i1/254593040/TB2NFT3apXXXXXkXpXXXXXXXXXX_!!254593040.jpg_30x30.jpg) center no-repeat;">
<span>草绿裸杯</span>

[extra]
tmall
<a href="#"><img src="http://gi4.md.alicdn.com/imgextra/i4/653392567/TB2NPDkaXXXXXXYXXXXXXXXXXXX_!!653392567.jpg_60x60q90.jpg" /></a>
taobao
 <a href="#"><img data-src="http://img04.taobaocdn.com/imgextra/i4/254593040/TB2B38VaVXXXXX4XXXXXXXXXXXX_!!254593040.jpg_50x50.jpg" /></a>
'''

