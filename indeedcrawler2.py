import requests
from bs4 import BeautifulSoup

l=[]
o={}


target_url = "https://www.indeed.com/jobs?q=python&l=New+York%2C+NY&vjk=8bf2e735050604df"

head= {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Connection": "keep-alive",
    "Accept-Language": "en-US,en;q=0.9,lt;q=0.8,et;q=0.7,de;q=0.6",
}

resp = requests.get(target_url)
print(resp.status_code)
soup = BeautifulSoup(resp.text, 'html.parser')

allData = soup.find("div",{"class":"mosaic-provider-jobcards"})

alllitags = allData.find_all("li",{"class":"eu4oa1w0"})
print(len(alllitags))
for i in range(0,len(alllitags)):
    try:
        o["name-of-the-job"]=alllitags[i].find("a").find("span").text
    except:
        o["name-of-the-job"]=None

    try:
        o["name-of-the-company"]=alllitags[i].find("span",{"data-testid":"company-name"}).text
    except:
        o["name-of-the-company"]=None



    try:
        o["job-location"]=alllitags[i].find("div",{"data-testid":"text-location"}).text
    except:
        o["job-location"]=None

    try:
        o["job-details"]=alllitags[i].find("div",{"class":"jobMetaDataGroup"}).text
    except:
        o["job-details"]=None

    l.append(o)
    o={}


print(l)