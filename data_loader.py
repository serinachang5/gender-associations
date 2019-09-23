import os

PROF_PATH = '../data/professor/'  # relative path to the professor data folder
CELEB_PATH = '../data/celeb/'  # relative path of the celeb data folder

'''
    This class parses the text files in the celebrity dataset (all three subfolders:
    eonline, people, and usweekly). During parsing, the celeb articles are divided into
    those that are labeled as female, as male, and as unknown. Each subset is stored in
    the form of a dictionary: the keys are the data ids and the values are various
    information about the article. This information includes the article's title,
    author(s), timestamp, URL, tag(s), predicted gender label, and text.
'''
class CelebDataLoader:
    def __init__(self, path_to_corpus, min_samples=None, verbose=False):
        self.path_to_corpus = path_to_corpus
        self.verbose = verbose
        print('Initializing CelebDataLoader...')
        self.female_corpus = {}
        self.male_corpus = {}
        self.unk_corpus = {}
        self._load_data(self.path_to_corpus + 'eonline/', min_samples)
        self._load_data(self.path_to_corpus + 'people/', min_samples)
        self._load_data(self.path_to_corpus + 'usweekly/', min_samples)

    def _load_data(self, txt_dir, min_samples):
        all_files = os.listdir(txt_dir)
        for fn in all_files:
            if fn.endswith('.txt'):
                path_to_file = txt_dir + fn
                try:
                    parsed = self._parse_file(path_to_file)
                    article_id = fn.strip('.txt')
                    parsed['id'] = article_id
                    if parsed['label'] == 1:
                        self.female_corpus[article_id] = parsed
                    elif parsed['label'] == 0:
                        self.male_corpus[article_id] = parsed
                    else:
                        self.unk_corpus[article_id] = parsed
                    if min_samples is not None and len(self.female_corpus) >= min_samples and len(self.male_corpus) >= min_samples:
                        break
                except:
                    if self.verbose:
                        print('Failed on', fn)
        print('After parsing {} -> {} female, {} male, {} unk'.format(txt_dir, len(self.female_corpus), len(self.male_corpus), len(self.unk_corpus)))

    def _parse_file(self, path_to_file):
        LINE_KEY = {'title':0, 'author':1, 'ts':2, 'url':3, 'tags':5, 'label':6, 'text':8}
        parsed = {}
        with open(path_to_file, 'r') as f:
            lines = f.readlines()
            for key,line_num in LINE_KEY.items():
                if line_num < len(lines):
                    val = lines[line_num].strip()
                    if key == 'tags':
                        val = val.strip('TAGS: ').split(',')
                    elif key == 'label':
                        val = int(val.strip('LABEL: '))
                else:
                    val = None
                parsed[key] = val
        return parsed

    def get_female_ids(self):
        return sorted(self.female_corpus.keys())

    def get_female_entries(self):
        fids = self.get_female_ids()
        return [self.female_corpus[fid] for fid in fids]

    def get_male_ids(self):
        return sorted(self.male_corpus.keys())

    def get_male_entries(self):
        mids = self.get_male_ids()
        return [self.male_corpus[mid] for mid in mids]

'''
    This class parses the text files in the professor dataset. During parsing, the
    professor reviews are divided into those that are labeled as female, as male, and
    as unknown. Each subset is stored in the form of a dictionary: the keys are the
    data ids and the values are various information about the professor and their reviews.
    This information includes the professor's name, their school, their RateMyProfessors
    URL, their number of reviews, their predicted gender, and their reviews, each of
    which contains the review's rating, tag(s), and text.
'''
class ProfDataLoader:
    def __init__(self, path_to_corpus, min_samples=None, verbose=False):
        self.path_to_corpus = path_to_corpus
        self.verbose = verbose
        print('Initializing ProfDataLoader...')
        self._load_data(min_samples)

    def _load_data(self, min_samples):
        self.female_corpus = {}
        self.male_corpus = {}
        self.unk_corpus = {}
        all_files = os.listdir(self.path_to_corpus)
        num_text_files = 0
        num_male_reviews = 0
        num_female_reviews = 0
        for fn in all_files:
            if fn.endswith('.txt'):
                num_text_files += 1
                path_to_file = self.path_to_corpus + fn
                parsed = self._parse_file(path_to_file)
                teacher_id = fn.rstrip('.txt')
                parsed['id'] = teacher_id
                num_reviews = len(parsed['reviews'])
                if parsed['gender'] == 'F':
                    self.female_corpus[teacher_id] = parsed
                    num_female_reviews += num_reviews
                elif parsed['gender'] == 'M':
                    self.male_corpus[teacher_id] = parsed
                    num_male_reviews += num_reviews
                else:
                    self.unk_corpus[teacher_id] = parsed
                if min_samples and len(self.female_corpus) >= min_samples and len(self.male_corpus) >= min_samples:
                    break
        print('Finished parsing RMP corpus. {} files in total -> M: {} files, {} reviews, F: {} files, {} reviews'.format(num_text_files, len(self.male_corpus), num_male_reviews, len(self.female_corpus), num_female_reviews))

    def _parse_file(self, path_to_file):
        with open(path_to_file, 'r') as f:
            lines = f.readlines()
            name = lines[0].strip()
            school = lines[1].lstrip('School:').strip()
            url = lines[2].lstrip('URL:').strip()
            num_reviews = int(lines[3].lstrip('Num reviews:').strip())
            metadata = {'name':name, 'school':school, 'url':url, 'num_reviews':num_reviews}
            gender = lines[4].strip().lstrip('Gender: ')
            i = 6  # index of first review
            reviews = []
            while i+3 < len(lines) and lines[i].startswith('Review #'):  # traverse through info
                rating = lines[i+1].lstrip('Rating:').strip()
                tags = lines[i+2].lstrip('Tags:').strip()
                text = lines[i+3].lstrip('Text: ').strip()
                reviews.append((rating, tags, text))
                i += 5
            return {'reviews':reviews, 'metadata':metadata, 'gender':gender}

    def get_female_ids(self):
        return sorted(self.female_corpus.keys())

    def get_female_entries(self):
        fids = self.get_female_ids()
        return [self.female_corpus[fid] for fid in fids]

    def get_female_reviews(self):
        reviews = []
        for entry in self.get_female_entries():
            for rating, tags, text in entry['reviews']:
                reviews.append(text)
        return reviews

    def get_male_ids(self):
        return sorted(self.male_corpus.keys())

    def get_male_entries(self):
        mids = self.get_male_ids()
        return [self.male_corpus[mid] for mid in mids]

    def get_male_reviews(self):
        reviews = []
        for entry in self.get_male_entries():
            for rating, tags, text in entry['reviews']:
                reviews.append(text)
        return reviews

    def get_unk_reviews(self):
        reviews = []
        for entry in self.unk_corpus.values():
            for rating, tags, text in entry['reviews']:
                reviews.append(text)
        return reviews

'''
    This function tests the celebrity data loader and prints out basic statistics
    about the dataset.
'''
def get_celeb_stats():
    dl = CelebDataLoader(CELEB_PATH, verbose=True)
    num_f = len(dl.female_corpus)
    num_m = len(dl.male_corpus)
    num_unk = len(dl.unk_corpus)
    print('Num female articles:', num_f)
    print('Num male articles:', num_m)
    print('Num unk articles:', num_unk)
    print('Num gendered articles:', num_f + num_m)
    print('Total num articles:', num_f + num_m + num_unk)

'''
    This function tests the professor data loader and prints out basic statistics
    about the dataset.
'''
def get_prof_stats():
    dl = ProfDataLoader(PROF_PATH, verbose=True)
    num_f_profs = len(dl.female_corpus)
    num_f_reviews = len(dl.get_female_reviews())
    num_m_profs = len(dl.male_corpus)
    num_m_reviews = len(dl.get_male_reviews())
    num_unk_reviews = len(dl.get_unk_reviews())
    print('Num female profs:', num_f_profs)
    print('Num reviews for female profs:', num_f_reviews)
    print('Num male profs:', num_m_profs)
    print('Num reviews for male profs:', num_m_reviews)
    print('Num gendered reviews:', num_f_reviews + num_m_reviews)
    print('Num total reviews:', num_f_reviews + num_m_reviews + num_unk_reviews)

if __name__ == '__main__':
    get_celeb_stats()