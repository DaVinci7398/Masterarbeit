from bs4 import BeautifulSoup
import requests

def find_jobs():
    html_text = requests.get('https://de.indeed.com/jobs?q=Jurist&l=&from=searchOnHP&vjk=86df3793c2e8f5a7')
    soup = BeautifulSoup(html_text, 'lxml')
    jobs = soup.find_all('li', class_ = 'css-1ac2h1w eu4oa1w0')
    print(jobs)