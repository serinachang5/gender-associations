from bs4 import BeautifulSoup
import json
import pickle
import re
import requests

# Filenames of the pickle files with the People/UsWeekly/E!Online articles will be stored
PEOPLE_URLS_FNAME = 'people_urls.pkl'
USWEEKLY_URLS_FNAME = 'usweekly_urls.pkl'
EONLINE_URLS_FNAME = 'eonline_urls.pkl'

PEOPLE_SAMPLE = 'https://people.com/movies/benedict-cumberbatch-on-what-to-get-grown-men-for-christmas/'
USWEEKLY_SAMPLE = 'https://www.usmagazine.com/stylish/news/golden-globes-2019-michelle-yeohs-makeup-details/'
EONLINE_SAMPLE = 'https://www.eonline.com/news/1011002/pete-davidson-and-kate-beckinsale-reunite-and-show-pda-after-comedy-show'

# ========== SCRAPE URLS ==========
'''
    This function scrapes the URLs of all articles tagged 'movie-celebrities' on the
    People website. max_pages limits the number of pages to scrape; it is mostly used
    for testing.
'''
def people_scrape_urls(max_pages):
    urls = []
    for page_num in range(1, max_pages+1):
        print('PAGE #{}'.format(page_num))
        page_url = 'https://people.com/tag/movie-celebrities/?page=' + str(page_num)
        r = requests.get(page_url)
        soup = BeautifulSoup(r.content, 'html.parser')
        links = soup.find_all('a', attrs={'class':'category-page-item-image-link'})
        if len(links) == 0:
            print('Breaking loop on page {}'.format(page_num))
            break
        for link in links:
            print(link['href'])
            urls.append(link['href'])
    pickle.dump(urls, open(PEOPLE_URLS_FNAME, 'wb'))

'''
    This function scrapes the URLs of all articles tagged 'celebrity-news' on the
    UsWeekly website. max_urls limits the number of urls to scrape; it is mostly used
    for testing.
'''
def usweekly_scrape_urls(max_urls):
    urls = []
    page_num = 1
    while True:
        page_url = 'https://www.usmagazine.com/celebrity-news/' + str(page_num)
        r = requests.get(page_url)
        soup = BeautifulSoup(r.content, 'html.parser')
        links = soup.find_all('a', attrs={'class':'content-card-link'})
        print('Page {}: adding {} links'.format(page_num, len(links)))
        if len(links) == 0:
            print('No more links -> breaking loop')
            break
        for link in links:
            urls.append(link['href'])
            if len(urls) % 100 == 0: # save every 100
                print('SAVING...')
                pickle.dump((urls, page_num), open(USWEEKLY_URLS_FNAME, 'wb'))
        if len(urls) >= max_urls:
            print('Reached max links -> breaking loop')
            break
        page_num += 1
    print('Found {} urls'.format(len(urls)))
    pickle.dump((urls, page_num), open(USWEEKLY_URLS_FNAME, 'wb'))

'''
    This function scrapes the URLs of all articles tagged 'news' on the E!Online website.
    min_page and max_page limit the number of pages to scrape; they are mostly used
    for testing.
'''
def eonline_scrape_urls(min_page, max_page, continue_work=True):
    assert(min_page <= max_page)
    if continue_work:
        urls = pickle.load(open(EONLINE_URLS_FNAME, 'rb'))
        print('Already have {} urls'.format(len(urls)))
    else:
        urls = []
    for page_num in range(min_page, max_page+1):
        print('PAGE #{}'.format(page_num))
        page_url = 'https://www.eonline.com/news/page/' + str(page_num)
        r = requests.get(page_url)
        soup = BeautifulSoup(r.content, 'html.parser')
        links = soup.find_all('a', attrs={'class':'category-landing__hero-link'})
        links += soup.find_all('a', attrs={'class':'category-landing__content-link'})
        if len(links) == 0:
            print('Breaking loop on page {}'.format(page_num))
            break
        for link in links:
            content_type = link.find('span', attrs={'class':'category-landing__textbox-type'})
            if content_type is None:
                content_type = link.find('div', attrs={'class':'content-item__type'})
            if content_type is not None:
                content_type = content_type.text.lower().strip()
                if content_type == 'news':
                    urls.append('https://www.eonline.com' + link['href'])
                    print(urls[-1])
                elif content_type != 'videos':
                    print(content_type)
    print('Done. Saving {} urls'.format(len(urls)))
    pickle.dump(urls, open(EONLINE_URLS_FNAME, 'wb'))

# ========== EXTRACT DATA ==========
'''
    This class extracts relevant data from a given soup, which is initialized by
    the article URL. The class is a wrapper for the corpus-specific extractors,
    PeopleExtractor, UsWeeklyExtractor, and EOnlineExtractor.
'''
class CelebExtractor:
    def __init__(self, dataset):
        if dataset == 'people':
            print('Making People Extractor')
            self._ext = PeopleExtractor()
        elif dataset == 'usweekly':
            print('Making UsWeekly Extractor')
            self._ext = UsWeeklyExtractor()
        elif dataset == 'eonline':
            print('Making E!Online Extractor')
            self._ext = EOnlineExtractor()
        else:
            raise Exception('Invalid dataset: ', dataset)

    def want_to_parse(self, soup):
        return self._ext.want_to_parse(soup)

    def extract_all(self, soup):
        return self._ext.extract_all(soup)

class PeopleExtractor:
    def __init__(self):
        self.BLACKLIST = {'RELATED: ',
                          'PEOPLE.com may receive compensation when you click',
                          'If you have opted in for our browser push notifications'}

    def want_to_parse(self, soup):
        return True

    def extract_all(self, soup):
        title = self.extract_title(soup)
        author = self.extract_author(soup)
        timestamp = self.extract_timestamp(soup)
        tags = self.extract_tags(soup)
        text = self.extract_text(soup)
        return title, author, timestamp, tags, text

    def extract_title(self, soup):
        title = soup.title
        if title is None:
            return 'NoTitle'
        return title.text.split(' | PEOPLE.com')[0]

    def extract_author(self, soup):
        author = soup.find('a', attrs={'class':'bold author-name'})
        if author is None:
            return 'NoAuthor'
        return author.text.strip()

    def extract_timestamp(self, soup):
        ts = soup.find('div', attrs={'class':'timestamp published-date'})
        if ts is None:
            return 'NoTimestamp'
        return ts.text.strip()

    def extract_tags(self, soup):
        return [t.text for t in soup.find_all('a', attrs={'class':'tag-link'})]

    def extract_text(self, soup):
        paragraphs = []
        for t in soup.find_all('p'):
            processed = self._preprocess_text(t.text)
            if processed is not None:
                paragraphs.append(processed)
        return paragraphs

    def _preprocess_text(self, text):
        text = text.strip().replace('\xa0', ' ')  # indicates link
        if any(text.startswith(s) for s in self.BLACKLIST):
            return None
        return text

class UsWeeklyExtractor:
    def __init__(self):
        self.ENDING_ADS = {'Sign up now for the Us Weekly newsletter',
                           'Want stories like these delivered straight to your phone?',
                           'Us Weekly has affiliate partnerships so we may receive compensation',
                           'Part of the American Media Inc. Celebrity News Network',
                           'More celebrity features on Yahoo!',
                           'For the latest celebrity entertainment, news and lifestyle videos'}

    def want_to_parse(self, soup):
        invalid_body = soup.find('body', attrs={'class':'single-format-gallery'})
        if invalid_body is None:
            return True
        return False

    def extract_all(self, soup):
        title = self.extract_title(soup)
        author = self.extract_author(soup)
        timestamp = self.extract_timestamp(soup)
        tags = self.extract_tags(soup)
        text = self.extract_text(soup)
        return title, author, timestamp, tags, text

    def extract_title(self, soup):
        title = soup.title
        if title is None:
            return 'NoTitle'
        return title.text.split(' - Us Weekly')[0]

    def extract_author(self, soup):
        author = soup.find('a', attrs={'rel':'author'})
        if author is None:
            return 'NoAuthor'
        return author.text.strip()

    def extract_timestamp(self, soup):
        ts = soup.find('meta', attrs={'property':'article:published_time'})
        if ts is None:
            return 'NoTimestamp'
        datetime = ts['content'].strip()
        print(datetime)
        date, time = datetime.split('T')
        year, month, date = date.split('-')
        monthname = monthnum_to_monthname(month)
        time = time.split('+',1)[0]
        hour, minute, second = time.split(':')
        if int(hour) <= 12:
            clock = 'AM'
        else:
            clock = 'PM'
        hour = int(hour) - 12
        return '{} {}, {} {}:{} {}'.format(monthname, date, year, hour, minute, clock)

    def extract_tags(self, soup):
        script_dict = self._find_script_dict(soup)
        if script_dict is not None and 'celebrity' in script_dict:
            if script_dict['celebrity'] == 'none':
                return []
            else:
                return script_dict['celebrity']
        return []

    def extract_text(self, soup):
        paragraphs = []
        for t in soup.find_all('p'):
            if t.find('a', attrs={'data-track-action':'Tap Interstitial Link'}) is None:
                processed = self._preprocess_text(t.text)
                if any(processed.startswith(s) for s in self.ENDING_ADS):
                    break
                else:
                    paragraphs.append(processed)
        return paragraphs

    def _find_script_dict(self, soup):
        script = soup.find(text=re.compile(r'utag_data'))
        if script is None:
            return None
        json_str = script.split(' = ', 1)[1].split('};', 1)[0] + '}'
        data = json.loads(json_str)
        return data

    def _preprocess_text(self, text):
        text = text.strip()
        text = text.replace('\xa0', ' ')  # indicates link
        text = text.replace('&apos', '\'')  # apostrophe
        return text

class EOnlineExtractor:
    def __init__(self):
        self.ENDING_ADS = {}

    def want_to_parse(self, soup):
        invalid_body = soup.find('body', attrs={'class':'single-format-gallery'})
        if invalid_body is None:
            return True
        return False

    def extract_all(self, soup):
        title = self.extract_title(soup)
        author = self.extract_author(soup)
        timestamp = self.extract_timestamp(soup)
        tags = self.extract_tags(soup)
        text = self.extract_text(soup)
        return title, author, timestamp, tags, text

    def extract_title(self, soup):
        title = soup.title
        if title is None:
            return 'NoTitle'
        return title.text.split(' | E! News')[0]

    def extract_author(self, soup):
        author = soup.find('span', attrs={'class':'entry-meta__author'})
        if author is None:
            return 'NoAuthor'
        text = author.text.strip()
        if text.startswith('by\n'):
            text = text.strip('by\n')
        toks = text.split()
        return ' '.join([t.capitalize() for t in toks])

    def extract_timestamp(self, soup):
        ts = soup.find('span', attrs={'class':'entry-meta__time'})
        if ts is None:
            return 'NoTimestamp'
        ts = ts.text.strip()
        day, datetime = ts.split(', ', 1)
        return datetime

    def extract_tags(self, soup):
        tags = []
        categories = soup.find_all('a', attrs={'class':'categories__link'})
        for cat in categories:
            text = cat.text.strip()
            if len(text) > 0:
                tags.append(text)
        return tags

    def extract_text(self, soup):
        paragraphs = []
        for t in soup.find_all('section', attrs={'data-textblock-tracking' : re.compile(r'.*')}):
            processed = self._preprocess_text(t.text)
            if any(processed.startswith(s) for s in self.ENDING_ADS):
                break
            else:
                paragraphs.append(processed)
        return paragraphs

    def _find_script_dict(self, soup):
        script = soup.find(text=re.compile(r'utag_data'))
        if script is None:
            return None
        json_str = script.split(' = ', 1)[1].split('};', 1)[0] + '}'
        data = json.loads(json_str)
        return data

    def _preprocess_text(self, text):
        text = text.strip()
        text = text.replace('\xa0', ' ')  # indicates link
        text = text.replace('&apos', '\'')  # apostrophe
        # addressing lack of spaces between sentences
        sents = text.split('.')
        for i, s in enumerate(sents):
            if i > 0 and len(s) > 0 and s[0].isalpha():
                sents[i] = ' ' + sents[i]
        return '.'.join(sents)

# ========== UTILS ==========
def monthname_to_monthnum(monthname):
    monthname = monthname.lower()
    if monthname.startswith('jan'):
        return 1
    if monthname.startswith('feb'):
        return 2
    if monthname.startswith('mar'):
        return 3
    if monthname.startswith('apr'):
        return 4
    if monthname.startswith('may'):
        return 5
    if monthname.startswith('jun'):
        return 6
    if monthname.startswith('jul'):
        return 7
    if monthname.startswith('aug'):
        return 8
    if monthname.startswith('sep'):
        return 9
    if monthname.startswith('oct'):
        return 10
    if monthname.startswith('nov'):
        return 11
    if monthname.startswith('dec'):
        return 12
    return None

def monthnum_to_monthname(monthnum):
    if type(monthnum) is str:
        monthnum = int(monthnum)
    if monthnum == 1:
        return 'January'
    if monthnum == 2:
        return 'February'
    if monthnum == 3:
        return 'March'
    if monthnum == 4:
        return 'April'
    if monthnum == 5:
        return 'May'
    if monthnum == 6:
        return 'June'
    if monthnum == 7:
        return 'July'
    if monthnum == 8:
        return 'August'
    if monthnum == 9:
        return 'September'
    if monthnum == 10:
        return 'October'
    if monthnum == 11:
        return 'November'
    if monthnum == 12:
        return 'December'
    return None

if __name__ == '__main__':
    ext = CelebExtractor('usweekly')
    r = requests.get(USWEEKLY_SAMPLE)
    soup = BeautifulSoup(r.content, 'html.parser')
    title, author, timestamp, tags, text = ext.extract_all(soup)
    print('TITLE:', title)
    print('AUTHOR:', author)
    print('TIME:', timestamp)
    print('TAGS:', tags)
    print('TEXT')
    for t in text:
        print(t)