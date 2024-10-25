import requests
from bs4 import BeautifulSoup
from random import choice
import time

# User-Agent rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; AS; rv:11.0) like Gecko',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
]

# Function to scrape job listings from Indeed
def scrape_indeed_jobs(search_term, location, num_pages=1):
    jobs = []
    
    for page in range(0, num_pages):
        # Construct URL for the given search term and location
        url = f'https://de.indeed.com/jobs?q={search_term}&l={location}&start={page * 10}'
        
        # Use a random User-Agent to mimic browser behavior
        headers = {'User-Agent': choice(USER_AGENTS)}
        
        # Send a request to the website
        response = requests.get(url, headers=headers)
        
        # If the request is successful, parse the HTML
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all job listings (job_seen_beacon is a class used for job listings on Indeed)
            job_cards = soup.find_all('div', class_='job_seen_beacon')
            
            for card in job_cards:
                # Extract job title
                job_title = card.find('h2').get_text(strip=True) if card.find('h2') else 'N/A'
                
                # Extract company name
                company = card.find('span', class_='companyName').get_text(strip=True) if card.find('span', class_='companyName') else 'N/A'
                
                # Extract job location
                location = card.find('div', class_='companyLocation').get_text(strip=True) if card.find('div', class_='companyLocation') else 'N/A'
                
                # Extract job summary
                summary = card.find('div', class_='job-snippet').get_text(strip=True) if card.find('div', class_='job-snippet') else 'N/A'
                
                # Save the job data
                jobs.append({
                    'Job Title': job_title,
                    'Company': company,
                    'Location': location,
                    'Summary': summary
                })
            
            # Sleep between requests to avoid being blocked
            time.sleep(choice(range(2, 5)))
        
        else:
            print(f"Failed to retrieve data from {url} (status code: {response.status_code})")
    
    return jobs

# Example usage: Scraping 2 pages of jurist jobs in Germany
search_term = "jurist"
location = "Deutschland"
num_pages = 2

scraped_jobs = scrape_indeed_jobs(search_term, location, num_pages)

# Print out the scraped job listings
for i, job in enumerate(scraped_jobs, 1):
    print(f"Job {i}:")
    print(f"Title: {job['Job Title']}")
    print(f"Company: {job['Company']}")
    print(f"Location: {job['Location']}")
    print(f"Summary: {job['Summary']}\n")
