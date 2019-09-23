from data_loader import CelebDataLoader, ProfDataLoader
from nltk import sent_tokenize
import pickle
import spacy

PATH_TO_CELEB_PREPROCESSED = '../preprocessed/celeb/'
PATH_TO_PROF_PREPROCESSED = '../preprocessed/professor/'

nlp = spacy.load('en_core_web_sm')

def make_celeb_toks_per_text(gender, continue_work=True):
    """
    Pre-processes the raw text data from the Celeb data loader.
    Two types of pre-processing are saved - at the article-level and at the
    sentence-level - and each pre-processed text is linked to the article ID
    that it came from. Saving article IDs also prevents repeating work (if
    continue_work is True).
    """
    if continue_work:
        old_toks_per_article = pickle.load(open(PATH_TO_CELEB_PREPROCESSED + '{}_toks_per_article.pkl'.format(gender), 'rb'))
        old_toks_per_sent = pickle.load(open(PATH_TO_CELEB_PREPROCESSED + '{}_toks_per_sent.pkl'.format(gender), 'rb'))
        old_article_ids = set([tuple[0] for tuple in old_toks_per_article])
    else:
        old_toks_per_article = []
        old_toks_per_sent = []
        old_article_ids = set()
    print('Already processed {} articles and {} sentences.'.format(len(old_toks_per_article), len(old_toks_per_sent)))
    articles = []
    article_ids = []
    for dataset in ['people', 'usweekly', 'eonline']:
        dl = CelebDataLoader(dataset)
        entries = dl.get_female_entries() if gender == 'f' else dl.get_male_entries()
        for e in entries:
            article_id = dataset + '_' + e['id']
            if article_id not in old_article_ids:
                articles.append(e['text'])
                article_ids.append(article_id)
    print('Processing {} new articles...'.format(len(articles)))
    toks_per_article, toks_per_sent, sent_ids = texts_to_pos_toks(article_ids, articles, verbose=True)
    print('Done! {} new articles, {} new sentences.'.format(len(toks_per_article), len(toks_per_sent)))
    new_toks_per_article = list(zip(article_ids, toks_per_article))
    pickle.dump(old_toks_per_article + new_toks_per_article, open(PATH_TO_CELEB_PREPROCESSED + '{}_toks_per_article.pkl'.format(gender), 'wb'))
    new_toks_per_sent = list(zip(sent_ids, toks_per_sent))
    pickle.dump(old_toks_per_sent + new_toks_per_sent, open(PATH_TO_CELEB_PREPROCESSED + '{}_toks_per_sent.pkl'.format(gender), 'wb'))

def make_prof_toks_per_text(gender, continue_work=True):
    """
    Pre-processes the raw text data from the Rate My Professor data loader.
    Two types of pre-processing are saved - at the review-level and at the
    sentence-level - and each pre-processed text is linked to the review ID
    that it came from. Saving review IDs also prevents repeating work (if
    continue_work is True).
    """
    dl = ProfDataLoader()
    if continue_work:
        old_toks_per_review = pickle.load(open(PATH_TO_PROF_PREPROCESSED + '{}_toks_per_review.pkl'.format(gender), 'rb'))
        old_toks_per_sent = pickle.load(open(PATH_TO_PROF_PREPROCESSED + '{}_toks_per_sent.pkl'.format(gender), 'rb'))
        old_review_ids = set([tuple[0] for tuple in old_toks_per_review])
    else:
        old_toks_per_review = []
        old_toks_per_sent = []
        old_review_ids = set()
    print('Already processed {} reviews and {} sentences.'.format(len(old_review_ids), len(old_toks_per_sent)))
    entries = dl.get_female_entries() if gender == 'f' else dl.get_male_entries()
    reviews = []
    review_ids = []
    for e in entries:
        teacher_id = e['id']
        for i, (rating, tags, text) in enumerate(e['reviews']):
            review_id = teacher_id + '#' + str(i)
            if review_id not in old_review_ids:
                reviews.append(text)
                review_ids.append(review_id)
    print('Processing {} new reviews...'.format(len(reviews)))
    toks_per_review, toks_per_sent, sent_ids = texts_to_pos_toks(review_ids, reviews, verbose=True)
    print('Done! {} new reviews, {} new sentences.'.format(len(toks_per_review), len(toks_per_sent)))
    new_toks_per_review = list(zip(review_ids, toks_per_review))
    pickle.dump(old_toks_per_review + new_toks_per_review, open(PATH_TO_PROF_PREPROCESSED + '{}_toks_per_review.pkl'.format(gender), 'wb'))
    new_toks_per_sent = list(zip(sent_ids, toks_per_sent))
    pickle.dump(old_toks_per_sent + new_toks_per_sent, open(PATH_TO_PROF_PREPROCESSED + '{}_toks_per_sent.pkl'.format(gender), 'wb'))

def texts_to_pos_toks(text_ids, texts, verbose=False):
    """
    Tokenizes sentences, then runs each sentence through a parser.
    Each token is represented by a tuple: <original_form, lemma, pos>
    """
    toks_per_text = []
    toks_per_sent = []
    sent_ids = []
    for i, (tid, text) in enumerate(zip(text_ids, texts)):
        text_toks = []
        sents = sent_tokenize(text)
        for sent in sents:
            sent_toks = _sent_to_pos_toks(sent)
            toks_per_sent.append(sent_toks)
            sent_ids.append(tid)
            text_toks += sent_toks
        toks_per_text.append(text_toks)
        if verbose and i % 1000 == 0:
            print(i)
    return toks_per_text, toks_per_sent, sent_ids

def _sent_to_pos_toks(sent):
    toks = []
    doc = nlp(sent)
    for tok in doc:
        pos = tok.pos_
        if pos != 'PUNCT':
            toks.append((tok.text, tok.lemma_, pos))
    return toks

if __name__ == '__main__':
    make_celeb_toks_per_text('f', continue_work=True)
    make_celeb_toks_per_text('m', continue_work=True)

    make_prof_toks_per_text('f', continue_work=True)
    make_prof_toks_per_text('m', continue_work=True)