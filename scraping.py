import os
from urllib import request as req
from urllib import error
from urllib import parse
import bs4

keyword ='サンタクロース'
dir_name = 'santa'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

urlKeyword = parse.quote(keyword)
url = 'https://www.google.com/search?hl=jp&q=' + urlKeyword + '&btnG=Google+Search&tbs=0&safe=off&tbm=isch'

headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0",}
request = req.Request(url=url, headers=headers)
page = req.urlopen(request)

html = page.read().decode('utf-8')
html = bs4.BeautifulSoup(html, "html.parser")
elems = html.select('.rg_meta.notranslate')
counter = 0
for ele in elems:
    ele = ele.contents[0].replace('"','').split(',')
    eledict = dict()
    for e in ele:
        num = e.find(':')
        eledict[e[0:num]] = e[num+1:]
    imageURL = eledict['ou']

    pal = '.jpg'
    if '.jpg' in imageURL:
        pal = '.jpg'
    elif '.JPG' in imageURL:
        pal = '.jpg'
    elif '.png' in imageURL:
        pal = '.png'
    elif '.gif' in imageURL:
        pal = '.gif'
    elif '.jpeg' in imageURL:
        pal = '.jpeg'
    else:
        pal = '.jpg'

    try:
        img = req.urlopen(imageURL)
        localfile = open('./'+dir_name+'/'+dir_name+str(counter)+pal, 'wb')
        localfile.write(img.read())
        img.close()
        localfile.close()
        counter += 1
    except UnicodeEncodeError:
        continue
    except error.HTTPError:
        continue
    except error.URLError:
        continue