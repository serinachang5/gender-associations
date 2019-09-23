from bs4 import BeautifulSoup
import os
import pickle
import requests
import numpy as np
from selenium import webdriver
from collections import Counter
import time

DOMAIN = 'https://www.ratemyprofessors.com'
PATH_TO_CORPUS = '../../corpus/'
COLUMBIA_ID = 278

def prep_query_by_school_driver():
    """
    Prepares a Chrome driver that puts the searches into query-by-school mode with the
    department set to Computer Science.
    """
    driver = webdriver.Chrome(os.path.join(os.getcwd(), 'chromedriver'))
    columbia_url = 'https://www.ratemyprofessors.com/search.jsp?queryBy=schoolId&schoolID={}&queryoption=TEACHER'.format(COLUMBIA_ID)
    driver.get(columbia_url)
    driver.find_element_by_class_name('close-this').click()
    dept_input = driver.find_element_by_xpath("//input[@placeholder='Enter Your Department']")
    dept_input.send_keys('Computer Science')
    cs_option = driver.find_element_by_xpath("//li[@data-value='Computer Science']")
    cs_option.click()
    return driver

def get_professors_from_school(driver, school_id, only_take_top_20 = False):
    """
    Gets the names and url's of professors for this school. If only_take_top_20,
    only the top (most reviewed) professors are included - this is easier because
    the top 20 are shown when the page loads. If all professors are desired, then
    the driver iterates through the alphabet and takes the top 20 for each
    filtered result (e.g. professor names starting with 'A'). This process usually
    gets all of the possible professors for the school, unless one school has more
    than 20 professors starting with one letter.
    """
    url = 'https://www.ratemyprofessors.com/search.jsp?queryBy=schoolId&schoolID={}&queryoption=TEACHER'.format(school_id)
    driver.get(url)
    num_professors = int(driver.find_element_by_xpath("//span[@class='professor-count']").text)
    if num_professors == 0:
        return num_professors, []
    if only_take_top_20 or num_professors < 20:
        return num_professors, get_current_list_of_professors(driver)
    results = []
    letter_filters = driver.find_elements_by_xpath("//a[@class='result']")
    for filter in letter_filters:
        filter_text = filter.text.strip()
        if filter_text != 'ALL':
            filter.click()
            time.sleep(.05)
            results += get_current_list_of_professors(driver)
    results = set(results)
    return num_professors, results

def get_current_list_of_professors(driver):
    """
    Gets the current professors listed on a school's page, given its filter settings.
    """
    results = []
    list_elems = driver.find_elements_by_xpath("//li[contains(@id, 'my-professor')]")
    for li in list_elems:
        link = li.find_element_by_tag_name('a')
        url = link.get_attribute('href')
        name = link.find_element_by_class_name('name').text.split('\n')[0]
        last, first = name.split(', ', 1)
        results.append((first + ' ' + last, url))
    return results

def extract_prof_id(url):
    """
    Given the url of a professor's page, return the Rate My Professor ID for
    this professor.
    """
    params = url.split('?', 1)[1].split('&')
    for p in params:
        key, value = p.split('=')
        if key == 'tid':
            return value
    return None

def parse_professor_page(url):
    """
    Parses the professor page and their reviews.
    """
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    reviews_heading = soup.find('div', attrs={'data-table':'rating-filter'})
    if reviews_heading is None:
        return 0, []
    num_reviews = int(reviews_heading.text.split()[0])
    reviews_table = soup.find('table', attrs={'class':'tftable'})
    reviews = []
    for row in reviews_table.find_all('tr')[1:]:
        if row.get('id'):
            reviews.append(_parse_reviews_row(row))
    return num_reviews, reviews

def _parse_reviews_row(row):
    """
    Helper function to parse one review object for its rating, tags,
    and text.
    """
    parsed = {}
    rating = row.find('span', attrs={'class':'rating-type'})
    if rating:
        parsed['rating'] = rating.text.strip()
    else:
        parsed['rating'] = None
    comments = row.find('td', attrs={'class':'comments'})
    if comments:
        tagbox = comments.find('div', attrs={'class':'tagbox'})
        if tagbox:
            tags = []
            for span_elem in tagbox.find_all('span'):
                tags.append(span_elem.text.strip())
            parsed['tags'] = tags
        else:
            parsed['tags'] = None
        paragraph = comments.find('p', attrs={'class':'commentsParagraph'})
        if paragraph:
            text = paragraph.text
            if text.startswith('"'):
                text.strip('"')
            if text.endswith('"'):
                text.strip('"')
            text = ' '.join(text.split())
            assert('\n' not in text)
            parsed['text'] = text
        else:
            parsed['text'] = None
    return parsed

def make_filename(prof_name, prof_url):
    """
    Makes the corpus filename from a professor's name and their page url.
    """
    tid = extract_prof_id(prof_url)
    prof_name_id = '_'.join(prof_name.split())
    return PATH_TO_CORPUS + '{}__{}.txt'.format(prof_name_id, tid)

MALE_PRONOUNS = ['he', 'him', 'his']
FEMALE_PRONOUNS = ['she', 'her', 'hers']

def predict_gender_from_reviews(reviews):
    """
    Predicts the gender of a professor, given their reviews.
    """
    m_count = 0
    f_count = 0
    for r in reviews:
        if r['text']:
            toks = r['text'].lower().split()
            counts = Counter(toks)
            for mp in MALE_PRONOUNS:
                if mp in counts:
                    m_count += counts[mp]
            for fp in FEMALE_PRONOUNS:
                if fp in counts:
                    f_count += counts[fp]
    if m_count > f_count:
        return 'M'
    if f_count > m_count:
        return 'F'
    return 'UNK'

def write_reviews_to_file(fn, prof_name, school_name, prof_url, num_reviews, gender, reviews):
    """
    Writes the information for a professor to file.
    """
    with open(fn, 'w') as f:
        f.write(prof_name + '\n')
        f.write('School: {}\n'.format(school_name))
        f.write('URL: {}\n'.format(prof_url))
        f.write('Num reviews: {}\n'.format(num_reviews))
        f.write('Gender: {}\n'.format(gender))
        f.write('\n')
        for i, rev in enumerate(reviews):
            f.write('Review #{}\n'.format(i+1))
            f.write('Rating: {}\n'.format(rev['rating']))
            f.write('Tags: {}\n'.format(', '.join(rev['tags'])))
            f.write('Text: {}\n'.format(rev['text']))
            f.write('\n')

def get_current_corpus():
    """
    Reviews all of the filenames in the current corpus.
    """
    corpus = set()
    for fn in os.listdir(PATH_TO_CORPUS):
        if fn.endswith('.txt'):
            corpus.add(PATH_TO_CORPUS + fn)
    return corpus

def prep_query_by_professor_driver():
    driver = webdriver.Chrome(os.path.join(os.getcwd(), 'chromedriver'))
    columbia_url = 'https://www.ratemyprofessors.com/search.jsp?queryBy=schoolId&schoolID={}&queryoption=TEACHER'.format(COLUMBIA_ID)
    driver.get(columbia_url)
    driver.find_element_by_class_name('close-this').click()
    return driver

def get_current_review_elems(driver):
    reviews_table = driver.find_element_by_class_name('tftable')
    rows = reviews_table.find_elements_by_tag_name('tr')[1:]  # first is always heading
    reviews = []
    for row in rows:
        if row.get_attribute('id'):  # only reviews (non-ads) have id
            reviews.append(row)
    return reviews

# --- EXEC ---
def collect_schools():
    """
    Collects the url's to all schools in the U.S. on Rate My Professor. Saved in
    school2id.pkl.
    """
    MIN_OFFSET = 0
    MAX_OFFSET = 6700
    STEP_SIZE = 20
    school2id = {}
    num_failed = 0
    for offset in np.arange(MIN_OFFSET, MAX_OFFSET+STEP_SIZE, step=STEP_SIZE):
        if offset % 100 == 0: print(offset)
        url = DOMAIN + '/search.jsp?query=&queryoption=HEADER&stateselect=&country=united+states&dept=&queryBy=schoolName&facetSearch=&schoolName=&offset={}&max=20'.format(offset)
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')
        schools = soup.find_all('li', attrs={'class':'listing SCHOOL'})
        for s in schools:
            try:
                link = s.find('a')
                school_id = int(link['href'].split('=')[-1])
                name = link.find('span', attrs={'class':'listing-name'}).find('span', attrs={'class':'main'}).text
                school2id[name] = school_id
            except:
                print('Failed:', s.text.strip())
                num_failed += 1
    print('Num schools found:', len(school2id))
    for s in school2id:
        if 'Columbia' in s:
            print(s, school2id[s])
    pickle.dump(school2id, open('../rate_my_prof/school2id.pkl', 'wb'))

def collect_professors_per_school(only_take_top_20):
    """
    Collects the list of CS professor pages per school. Saved in school2info.pkl.
    """
    school2id = pickle.load(open('../rate_my_prof/school2id.pkl', 'rb'))
    sorted_schools = sorted(list(school2id.keys()))
    print(len(sorted_schools))
    school2info = {}
    driver = prep_query_by_school_driver()
    total_num_profs = 0
    total_num_prof_pages = 0
    for i, school in enumerate(sorted_schools):
        try:
            sid = school2id[school]
            num_profs, prof_pages = get_professors_from_school(driver, sid, only_take_top_20=only_take_top_20)
            total_num_profs += num_profs
            total_num_prof_pages += len(prof_pages)
            school = school.strip()
            school2info[school] = (sid, num_profs, prof_pages)
            pickle.dump(school2info, open('../rate_my_prof/school2info.pkl', 'wb'))
            print('{}. School: {}. Num CS profs: {} -> SUCCESS'.format(i, school, num_profs, len(prof_pages)))
        except Exception as e:
            print('{}. School: {} -> FAILED'.format(i, school), e)
    driver.quit()
    print('Processed {} schools'.format(len(school2info)))
    print('{} CS profs in total'.format(total_num_profs))
    print('{} prof pages collected'.format(total_num_prof_pages))

def edit_professors_per_school():
    """
    Edits school2info.pkl to collect more professor pages for schools with
    more than 20 CS professors.
    """
    driver = prep_query_by_school_driver()
    fn = '../1.rate_my_prof/school2info.pkl'
    school2info = pickle.load(open(fn, 'rb'))
    missing_before = 0
    missing_now = 0
    for school, (sid, num_profs, prof_pages) in school2info.items():
        if len(prof_pages) < num_profs:
            missing_before += num_profs - len(prof_pages)
            try:
                num_profs, prof_pages = get_professors_from_school(driver, sid, only_take_top_20=False)
                print('{} -> got {} out of {}'.format(school, len(prof_pages), num_profs))
                missing_now += num_profs - len(prof_pages)
                school2info[school] = (sid, num_profs, prof_pages)
            except:
                print('Failed parsing {} -> no change'.format(school))
                missing_now += num_profs - len(prof_pages)  # still missing same amount
    print('Missing {} profs before, missing {} profs now'.format(missing_before, missing_now))
    pickle.dump(school2info, open(fn, 'wb'))

def build_corpus(start_idx, num_schools_to_process):
    """
    Builds the text corpus, where there is one text file per professor, and the
    text file consists of all of that professor's reviews.
    """
    current_corpus = get_current_corpus()
    school2info = pickle.load(open('../1.rate_my_prof/school2info.pkl', 'rb'))
    sorted_schools = sorted(list(school2info.keys()))
    print('Total num schools:', len(sorted_schools))
    end_idx = min(len(sorted_schools), start_idx + num_schools_to_process)
    print('Processing schools from idx {} to {} ({} schools)'.format(start_idx, end_idx-1, end_idx-start_idx))
    total_num_new_reviews = 0
    for i in range(start_idx, end_idx):
        school = sorted_schools[i]
        sid, num_profs, prof_pages = school2info[school]
        if len(prof_pages) == 0:
            print('{}. {} -> no data on CS professors'.format(i, school))
        else:
            school_num_new_reviews = 0
            for prof_name, prof_url in prof_pages:
                fn = make_filename(prof_name, prof_url)
                if fn not in current_corpus:
                    try:
                        num_reviews, processed_reviews = parse_professor_page(prof_url)
                        if len(processed_reviews) > 0:
                            gender = predict_gender_from_reviews(processed_reviews)
                            write_reviews_to_file(fn, prof_name, school, prof_url, num_reviews, gender, processed_reviews)
                            school_num_new_reviews += len(processed_reviews)
                            total_num_new_reviews += len(processed_reviews)
                    except:
                        print('Warning: failed on Prof. {} (id:{})'.format(prof_name, extract_prof_id(prof_url)))
            print('{}. {} -> num prof pages = {}, num new reviews = {}'.format(i, school, len(prof_pages), school_num_new_reviews))
    print('\nFINISHED!')
    new_corpus = get_current_corpus()
    print('Num profs before: {}. Num profs now: {}.'.format(len(current_corpus), len(new_corpus)))


if __name__ == '__main__':
    # collect_schools()
    # collect_professors_per_school(only_take_top_20=True)
    build_corpus(5170, 5000)
    # edit_professors_per_school()