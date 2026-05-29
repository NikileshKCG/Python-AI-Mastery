"""
Project 5: Web Scraper
Concepts: requests, BeautifulSoup, REST APIs, JSON parsing,
          regex, rate limiting, headers, pagination,
          data export (CSV + JSON), error handling
"""

import requests               # HTTP requests
from bs4 import BeautifulSoup       # HTML parsing
import json
import csv
import re                           # regular expressions
import time
import os
from datetime import datetime
from urllib.parse import urljoin, urlparse  # URL utilities 

#--------------------------------------------------------------
# SECTION 1: HTTP Session with Headers
# Concepts: requests.Session reuses connection
#           headers mimic a real browser to avoid blocks
#--------------------------------------------------------------

def create_session():
    """ Create a reusable HTTP session with browser-like headers. """
    session = requests.Session()
    session.headers.update({
        "User-Agent"        : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) " 
                              "Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language"   : "en-US,en;q=0.9",
        "Accept-Encoding"   : "gzip, deflate, br",
        "Accept"            : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    return session

def fetch_page(url, session=None, timeout=10, retries=3):
    """
    Fetch a web page with retry logic.
    Concept: retry loop, timeout, status code checking
    """
    if session is None:
        session = create_session()
    
    for attempt in range(1, retries + 1):
        try:
            print(f" Fetching: {url} (attempt {attempt})")
            response = session.get(url, timeout=timeout)
            response.raise_for_status()         # raises exception for 4xx/5xx status codes
            return response
        except requests.exceptions.HTTPError as e:
            print(f" HTTP Error: {e}")
        except requests.exceptions.ConnectionError:
            print(f" Connection error, Retrying.....")
        except requests.exceptions.Timeout:
            print(f" Request timed out.")
        except requests.exceptions.RequestException as e:
            print:(f" Request failed: {e}")
        
        if attempt < retries:
            time.sleep(2 ** attempt)                # exponential backoff: 2, 4, 8 seconds
    
    print(f" Failed to fetch after {retries} attempts. ")
    return None

#--------------------------------------------------------------
# SECTION 2: HTML Parser Utilities
# Concepts: BeautifulSoup - navingating the DOM tree
#--------------------------------------------------------------

def parse_html(html_content):
    """ Parse raw HTML into a BeautifulSoup object. """
    return BeautifulSoup(html_content, "html.parser")

def safe_text(element):
    """ Extract text from a BS4 element safely (returns '' if None). """
    return element.get_text(strip=True) if element else ""

def safe_attr(element, attr):
    """ Extract an attribute from a BS4 element safely. """
    return element.get(attr, "") if element else ""

#--------------------------------------------------------------
# SECTION 3: Scraper 1 - Quotes (quotes.toscrapre.com)
# Concepts: find(), find_all(), CSS selectors,
#           pagination, data extraction
#--------------------------------------------------------------

def scarpe_quotes(max_pages=3):
    """
    Scrape quotes, authors, and tags from quotes .toscrape.com
    A beginner-friendly site made specifically for scraping practice.
    """
    base_url = "https://quotes.toscrape.com"
    session = create_session()
    all_quotes = []
    page = 1

    print(f"\n Scraping Quotes (up to {max_pages} pages)....\n")

    while page <= max_pages:
        url      = f"{base_url}/page/{page}"
        response = fetch_page(url, session)
        if not response:
            break

        soup = parse_html(response.text)

        # find_all() returns a list of all matching tags 
        quote_blocks = soup.find_all("div", class_="quote")

        if not quote_blocks:
            # CSS selector: .text, .author -- like jQuery selectors
            text   = safe_text(block.select_one(".text"))
            author = safe_text(block.select_one(".author"))

            # find_all returns a list -- list comprehension extracts text from each
            tags = [safe_text(tag) for tag in block.find_all("a", class_="tag")]

            all_quotes.append({
                "quote"  : text,
                "author" : author,
                "tags"   : ", ".join(tags),
                "page"   : page
            })
        
        print(f" Page {page}: {len(quote_blocks)} quotes collected. ")
        page += 1
        time.sleep(1)                   # polite delay between requests
    
    print(f"\n Total quotes collected: {len(all_quotes)}")
    return all_quotes

#--------------------------------------------------------------
# SECTION 4: Scraper 2 - Books (books.toscrape.com)
# Concepts: Extracting structured data, star ratings,
#           relative URL resolution, urljoin
#--------------------------------------------------------------

STAR_MAP = { "One": 1, "Two": 2, "Three": 3, "Four":4, "Five": 5}

def scrape_books(max_pages=3):
    """
    Scrape book titles, prices, ratings, availablility
    from books.toscrape.com -- another scraping practice site.
    """
    base_url  = "https://books.toscrapre.com/catalogue/"
    session   = create_session()
    all_books = []
    url       = "https://books.toscrapre.com/catalogue/page-1.html"
    page      = 1

    print(f"\n Scraping Books (up to {max_pages} pages).... \n")

    while page <= max_pages:
        response = fetch_page(url, session)
        if not response:
            break

        soup    = parse_html(response.text)
        books   = soup.find_all("article", class_="product_pod")

        for book in books:
            title = safe_attr(book.find("h3").find("a"), "title")

            # Price -- Strip currency symbol 
            price_raw = safe_text(book.find("p", class_="price_color"))
            price = re.sub(r"[^\d.]", "", price_raw)        # regex: remove non-digit/dot chars

            # Rating stored in a CSS class name e.g. "star-rating Three"
            rating_tag  = book.find("p", class_="star-rating")
            rating_word = rating_tag["class"][1] if rating_tag else "Zero"
            rating      = STAR_MAP.get(rating_word, 0)

            #Availability
            availability = sage_text(book.find("p", class_="instock"))

            # Thumbnail URL - urljoin resolves relative URLs to absolute
            img_src = safe_attr(book.find("img"), "src")
            img_url = urljoin("https://books.toscrape.com/", img_src)

            all_books.append({
                "title"         : title,
                "price_gbp"     : price,
                "rating_stars"  : rating,
                "availability"  : availability,
                "image_url"     : img_url,
                "page"          : page
            })
        
        print(f" Page {page}: {len(books)} books collected.")

        # Find "next" button and follow it
        next_btn = soup.find("li", class_="next")
        if not next_btn or page >= max_pages:
            break

        next_href = safe_attr(next_btn.find("a"), "href")
        url       = urljoin(base_url, next_href)  # resolve relative URL
        page     += 1
        time.sleep(1)

    print(f"\n Total books collected: {len(all_books)}")
    return all_books

#--------------------------------------------------------------
# SECTION 5: REST API Scraper -- Public APIs
# Concepts: JSON APIs, response.json(), query params,
#           nested JSON parsing, API pagination
#--------------------------------------------------------------

def fetch_github_repos(username):
    """
    Fetch public Github repositories for any username.
    Uses Github's free public API -- no key needed.
    """
    url     = f"https://api.github.com/users/{username}/repos"
    params  = {"per_page": 30, "sort": "updated"}   # query parameters
    session = create_session()
    session.headers.update({"Accept": "application/vnd.github.v3+json"})

    print(f"\n Fetching Github repos for: {username} \n")

    response = fetch_page(url + "?" + "&".join(f"{k}={v}" for k, v in params.items()), session)
    if not response:
        return []
    
    repos_raw = response.json()

    if isinstance(repos_raw, dict) and "message" in repos_raw:
        print(f" Github API: {repos_raw['message']}")
        return []
    
    repos = []
    for repo in repos_raw:
        repos.append({
            "name"          : repo.get("name", ""),
            "description"   : repo.get("description", "") or "No description",
            "language"      : repo.get("language", "") or "N/A",
            "stars"         : repo.get("stargazers_count", 0),
            "forks"         : repo.get("forks_count", 0),
            "url"           : repo.get("html_urls", ""),
            "updated_at"    : repo.get("updated_at", ""),  # slice date from datetime string
        })
    
    # Sort by stars descending - sorted() with Lambda
    repos = sorted(repos, key=lambda r: r["stars"], reverse=True)

    print(f" {len(repos)} repositories found.. \n")
    for r in repos[:10]:            # show top 10
        print(f" {r['stars']:>5} Forks {r['forks']:>4} [{r['language']:<12}] {r['name']}")
    
    return repos

def fetch_public_api_data(api_choice):
    """
    Fetch data from various free public APIs.
    Demonstrates different JSON structures.
    """
    apis = {
        "1": {
            "name"   : "Random Joke",
            "url"    : "https://official-joke-api.appspot.com/random_joke",
            "parser" : lambda d: f"\n :D {d['setup']}\n  {d['punchline']}"
        },
        "2": {
            "name"   : "Random Dog Image",
            "url"    : "https://dog.ceo/api/breeds/image/random",
            "parser" : lambda d: f"\n Dog image URL: {d["message"]}"
        },
        "3": {
            "name"   : "Random Advice",
            "url"    : "https://api.advicelip.com/advice",
            "parser" : lambda d: f"\n Advice #{d['slip']['id']}: {d['slip']['advice']}"
        },
        "4": {
            "name"   : "ISS Current Location",
            "url"    : "https://api.open-notify.org/iss-now.json",
            "parser" : lambda d: (
                f"\n ISS Location: \n"
                f" Latitude  : {d['iss_position']['latitude']}\n"
                f" Longitude : {d['iss_position']['longitude']}"
            )
        },
        "5": {
            "name"   : "Random User Profile",
            "url"    : "https://randomuser.me/api",
            "parser" : lambda d: (
                lambda u: (
                    f"\n {u['name']['first']} {u['name']['last']}\n"
                    f" Email    : {u['email']}\n"
                    f" Country  : {u['location']['country']}\n"
                    f" Phone    : {u['phone']}"
                )
            )(d["results"][0])
        },
    }

    if api_choice not in apis:
        print(f" Invalid Choice. ")
        return
    
    api      = apis[api_choice]
    session  = create_session()
    response = fetch_page(api["url"], session)

    if not response:
        return
    
    data    = response.json()
    output  = api["parser"](data)
    print(f" {api['name']}:{output}")
    return data

#--------------------------------------------------------------
# SECTION 6: Link Extractor
# Concepts: find_all("a"), href filtering, urlparse
#           ses for deduplication
#--------------------------------------------------------------

def extract_links(url, internal_only=False):
    """ Extract all hyperlinks from a webpage. """
    session = create_session()
    response = fetch_page(url, session)
    if not response:
        return []
    
    soup    = parse_html(response.text)
    domain  = urlparse(url).netloc          # extract domain from URL

    links = set()                               # set automatically deduplications
    for tag in soup.find_all("a", href=True):   # href=True filters only tags with href
        href= tag["href"].strip()

        # Skip anchors, javascript, mailto
        if href.startswith(('#', "javascript:", "mailto:")):
            continue

        # Resolve relative URLs
        full_url = urljoin(url, href)

        if internal_only:
            if urlparse(full_url).netloc == domain:
                links.add(full_url)
        else:
            links.add(full_url)

    sorted_links = sorted(links)
    print(f"\n Found {len(sorted_links)} links on {url}:\n")
    for link in sorted_links[:30]:          # show first 30
        print(f" {link}")

    return sorted_links

#--------------------------------------------------------------
# SECTION 7: Regex Extractor
# Concepts: re module -- finding patterns in raw text
#--------------------------------------------------------------

def extract_with_regex(url):
    """ Etract emails, phone numbers and dates from a page using regex. """
    session     = create_session()
    response    = fetch_page(url, session)
    if not response:
        return
    
    text = response.text

    # Regex patterns -- compile once, use many times (faster)
    email_pattern = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    phone_pattern = re.compile(r"\+?[\d\s\-().]{10,15}")
    date_pattern  = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")

    emails = set(email_pattern.findall(text))
    phones = set(phone_pattern.findall(text))
    dates  = set(date_pattern.findall(text))

    print(f"\n Regex Extraction from: {url}")
    print(f"\n Email found ({len(emails)}):")
    for e in list(emails)[:10]:
        print(f" {e}")
    
    print("\n Dates found ({len(dates)}):")
    for d in list(dates)[:10]:
        print(f" {d}")

#--------------------------------------------------------------
# SECTION 8: Data Export
# Concepts: csv.DictWriter, json.dump, file naming
#--------------------------------------------------------------

def export_to_json(data, filename=None):
    """ Export scraped data to a JSON file."""
    if not data:
        print(" No data to export..")
        return
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%D_%H%M%S")
        filename  = f"scraped_{timestamp}.json"
    
    with open (filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f" Exported {len(data)} records to '{filename}'")

def export_to_csv(data, filename=None):
    """ Export scraped data (list of dicts) to a CSV file."""
    if not data:
        print(" No data to export..")
        return
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%D_%H%M%S")
        filename  = f"scraped_{timestamp}.csv"
    
    fieldnames = data[0].keys()             # column headers from first dict's keys
    
    with open (filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()            # write column names
        writer.writerows(data)          # write all rows at once

    print(f" Exported {len(data)} records to '{filename}'")

#----------------------------------------------------------
# SECTION 9: Main Menu
#-----------------------------------------------------------

def main():
    print("\n" + "="* 48)
    print(" WEB SCRAPER & API EXPLORER ")
    print("="*48)

    last_data = []

    while True:
        print("""
---------------- MAIN MENU ------------------------------------
              [1]  Scrape Quotes (quotes.toscrape.com)
              [2]  Scrape Books (books.toscrape.com)
              [3]  GitHub Repos (public API)
              [4]  Public APIs (jokes/dogs/advice/ISS)
              [5]  Extract Links from any URL
              [6]  Regex Extractor (emails, dates)
              [7]  Export last result --> JSON
              [8]  Export last result --> CSV
              [0]  Exit
-------------------------------------------------------------""")
        
        choice = input(" Enter Choice: ").strip()

        try:
            if choice == "1":
                pages = int(input(" Pages to scrape (1-5): ").strip() or "2")
                last_data = scarpe_quotes(max_pages=min(pages, 5))
            
            elif choice == "2":
                pages = int(input(" Pages to scrape (1-5): ").strip() or "2")
                last_data = scrape_books(max_pages=min(pages, 5))
            
            elif choice == "3":
                username = input(" GitHub Username: ").strip()
                last_data = fetch_github_repos(username)
            
            elif choice == "4":
                print("""
                      [1] Random Joke
                      [2] Random Dog Iamge
                      [3] Random Advice
                      [4] ISS Current Location
                      [5] Random User Profile
--------------------------------""")
                api_choice = input(" Choose API: ").strip()
                fetch_public_api_data(api_choice)
            
            elif choice == "5":
                url       = input(" Enter URL: ").strip()
                internal  = input(" Internal links only? (yes/no): ").strip().lower()
                last_data = [{"url": u} for u in extract_links(url, internal == "yes")]
            
            elif choice == "6":
                url = input(" Enter URL: ").strip()
                extract_with_regex(url)
            
            elif choice == "7":
                fname = input(" Filename (blank for auto): ").strip() or None
                export_to_json(last_data, fname)
            
            elif choice == "8":
                fname = input(" Filename (blank for auto): ").strip() or None
                export_to_csv(last_data, fname)
            
            elif choice == "0":
                print("\n GoodBye!!!\m")
                break

            else:
                print(" Invalid Choice, try again!")
        
        except ValueError as e:
            print(f" Value Error: {e}")
        except Exception as e:
            print(f" Error: {e}")

if __name__ == "__main__":
    main()