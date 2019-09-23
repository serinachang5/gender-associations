from bs4 import BeautifulSoup
from celeb_extractor import CelebExtractor, monthname_to_monthnum
from collections import Counter
import os
import pickle
import requests

class CelebBuilder:
    def __init__(self, dataset, verbose=False):
        self.ext = CelebExtractor(dataset)
        self.verbose = verbose

    def want_to_parse(self, soup):
        return self.ext.want_to_parse(soup)

    def write_to_file(self, dir, url, soup):
        title, author, timestamp, tags, text = self.ext.extract_all(soup)
        ID = self._make_id(author, timestamp)
        print('ID: {} | Title: {}'.format(ID, title))
        filename = dir + '{}.txt'.format(ID)
        with open(filename, 'w') as f:
            f.write('{}\n'.format(title))
            f.write('{}\n'.format(author))
            f.write('{}\n'.format(timestamp))
            f.write('{}\n\n'.format(url))
            f.write('TAGS: {}\n'.format(', '.join(tags)))
            if len(tags) > 0:
                label = self._determine_label(tags)
            else:
                label = -1
            f.write('LABEL: {}\n\n'.format(str(label)))
            f.write(' '.join(text))

    def _make_id(self, author, timestamp):
        last_name = author.split()[-1]
        monthname, date, year, timestamp, clock = timestamp.split()
        if monthname.endswith('.'):
            monthname = monthname[:-1]
        if date.endswith(','):
            date = date[:-1]
        month = str(monthname_to_monthnum(monthname))
        if len(month) == 1:
            month = '0' + month
        timestamp = timestamp.replace(':','.')
        return '{}-{}-{}_{}{}_{}'.format(year, month, date, timestamp, clock, last_name)

    def _determine_label(self, tags):
        fcount = 0
        mcount = 0
        for t in tags:
            soup, found_bio = self._find_wiki(t)
            if found_bio:
                summary = soup.find_all('p', limit=4)[1:]  # first is blank
                gen = predict_gender([p.text for p in summary])
                if gen == 'F':
                    fcount += 1
                elif gen == 'M':
                    mcount += 1
        if fcount > 0 and mcount == 0:
            return 1
        if mcount > 0 and fcount == 0:
            return 0
        return -1

    def _find_wiki(self, text):
        text = text.split()
        r = requests.get('https://en.wikipedia.org/w/index.php?search=' + '+'.join(text))
        soup = BeautifulSoup(r.content, 'html.parser')
        page_name = soup.find('h1', attrs={'class':'firstHeading'}).text
        found_bio = False
        report = '{} -> Wiki page \'{}\' -> '.format(text, page_name)
        if page_name != 'Search results':
            rows = soup.find_all('th', attrs={'scope':'row'})
            row_labels = [r.text for r in rows]
            if 'Born' in row_labels:
                report += 'FOUND BIO'
                found_bio = True
            else:
                report += 'IGNORE (NOT BIO)'
        else:
            report += 'IGNORE (SEARCH RESULTS)'
        if self.verbose:
            print(report)
        return soup, found_bio

def predict_gender(texts):
    fcount = 0
    mcount = 0
    for text in texts:
        text = text.lower().split()
        ct = Counter(text)
        fcount += ct['she']
        fcount += ct['her']
        mcount += ct['he']
        mcount += ct['his']
    if fcount > mcount:
        return 'F'
    if mcount > fcount:
        return 'M'
    return 'UNK'

def make_corpus(dataset, startover=False, max_to_process=None):
    builder = CelebBuilder(dataset=dataset)
    urls_fn = dataset + '_urls.pkl'
    urls_to_process = pickle.load(open(urls_fn, 'rb'))
    failed_fn = dataset + '_failed_urls.pkl'
    skipped_fn = dataset + '_skipped_urls.pkl'
    if startover:
        failed = set()
        skipped = set()
    else:
        failed = pickle.load(open(failed_fn, 'rb'))
        skipped = pickle.load(open(skipped_fn, 'rb'))
    text_dir = '../../data/celeb/{}/'.format(dataset)
    print('Saving files in {}'.format(text_dir))
    if startover:
        num_parsed = 0
        num_processed = 0
    else:
        num_parsed = get_num_parsed(text_dir)
        num_processed = num_parsed + len(failed) + len(skipped)
    urls_to_process = urls_to_process[num_processed:]
    if max_to_process:
        urls_to_process = urls_to_process[:max_to_process]
    print('Num already processed: {}. Num to process: {}'.format(num_processed, len(urls_to_process)))

    for i, url in enumerate(urls_to_process):
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')
        if not builder.want_to_parse(soup):  # quick check of whether this type of page should be parsed
            print('Skipping:', url)
            skipped.add(url)
        else:
            try:
                builder.write_to_file(text_dir, url, soup)
                num_parsed += 1
            except ValueError:
                print('Could not parse:', url)
                failed.add(url)
        num_done = i+1
        if num_done % 50 == 0:
            print('DONE WITH {} ARTICLES!'.format(num_done))
    pickle.dump(failed, open(failed_fn, 'wb'))
    pickle.dump(skipped, open(skipped_fn, 'wb'))
    print('Overall status: parsed {}, failed on {}, skipped {}'.format(num_parsed, len(failed), len(skipped)))

def get_num_parsed(text_dir):
    fns = [fn for fn in os.listdir(text_dir) if fn.endswith('.txt')]
    return len(fns)

if __name__ == '__main__':
    make_corpus('people', startover=False, max_to_process=300000)
