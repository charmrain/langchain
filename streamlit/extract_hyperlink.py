import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def is_valid_link(link):
    return link.startswith("http://") or link.startswith("https://")

def extract_hyperlinks(url):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if request was successful
    if response.status_code == 200:
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all anchor tags
        links = soup.find_all('a')
        
        # Extract href attribute from each anchor tag and validate
        hyperlinks = [urljoin(url, link.get('href')) for link in links if link.get('href') and is_valid_link(link.get('href'))]
        
        return hyperlinks
    else:
        print("Failed to fetch page:", response.status_code)
        return []

if __name__ == "__main__":
    # Example URL
    url = "https://www.performixbiz.com/"
    
    # Extract hyperlinks
    hyperlinks = extract_hyperlinks(url)
    
    # Print hyperlinks
    for link in hyperlinks:
        print(link)
    print("no. of links:",len(hyperlinks))
