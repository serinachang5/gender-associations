from collections import Counter
from nltk.corpus import stopwords
import pickle
from preprocessing import PATH_TO_CELEB_PROCESSED, PATH_TO_PROF_PROCESSED
from scipy.stats import beta

STOPWORDS = stopwords.words('english')

def get_tok_counts_from_balanced_celeb_corpus():
    """
    Loads the pre-processed articles and undersamples the larger one. Counts are then
    computed over the <lemma>,<pos> tuples in the kept articles.
    """
    f_toks_per_article_w_id = pickle.load(open(PATH_TO_CELEB_PROCESSED + 'f_toks_per_article.pkl', 'rb'))
    m_toks_per_article_w_id = pickle.load(open(PATH_TO_CELEB_PROCESSED + 'm_toks_per_article.pkl', 'rb'))
    f_toks_per_article = [tup[1] for tup in f_toks_per_article_w_id]
    m_toks_per_article = [tup[1] for tup in m_toks_per_article_w_id]
    print('Original lengths:', len(f_toks_per_article), len(m_toks_per_article))
    if len(f_toks_per_article) > len(m_toks_per_article):
        f_toks_per_article = f_toks_per_article[:len(m_toks_per_article)]
    elif len(m_toks_per_article) > len(f_toks_per_article):
        m_toks_per_article = m_toks_per_article[:len(f_toks_per_article)]
    print('Balanced lengths:', len(f_toks_per_article), len(m_toks_per_article))
    f_counts = compute_lemma_pos_counts(f_toks_per_article)
    m_counts = compute_lemma_pos_counts(m_toks_per_article)
    return f_counts, m_counts

def get_tok_counts_from_balanced_prof_corpus():
    """
    Loads the pre-processed reviews and undersamples the larger one. Counts are then
    computed over the <lemma>,<pos> tuples in the kept reviews.
    """
    f_toks_per_review_w_id = pickle.load(open(PATH_TO_PROF_PROCESSED + 'f_toks_per_review.pkl', 'rb'))
    m_toks_per_review_w_id = pickle.load(open(PATH_TO_PROF_PROCESSED + 'm_toks_per_review.pkl', 'rb'))
    f_toks_per_review = []
    m_toks_per_review = []
    for review_id, toks in f_toks_per_review_w_id:
        f_toks_per_review.append(toks)
    for review_id, toks in m_toks_per_review_w_id:
        m_toks_per_review.append(toks)
    print('Original lengths:', len(f_toks_per_review), len(m_toks_per_review))
    if len(f_toks_per_review) > len(m_toks_per_review):
        f_toks_per_review = f_toks_per_review[:len(m_toks_per_review)]
    elif len(m_toks_per_review) > len(f_toks_per_review):
        m_toks_per_review = m_toks_per_review[:len(f_toks_per_review)]
    print('Balanced lengths:', len(f_toks_per_review), len(m_toks_per_review))
    f_counts = compute_lemma_pos_counts(f_toks_per_review)
    m_counts = compute_lemma_pos_counts(m_toks_per_review)
    return f_counts, m_counts

def compute_lemma_pos_counts(toks_per_text):
    all_toks = []
    for toks in toks_per_text:
        for word, lemma, pos in toks:
            all_toks.append((lemma, pos))
    return Counter(all_toks)

def beta_scoring_from_counts(fcounts, mcounts, min_count=5):
    f_N = sum([count for count in fcounts.values()])
    m_N = sum([count for count in mcounts.values()])
    all_counts = fcounts + mcounts
    N = f_N + m_N
    f_associated = []
    m_associated = []
    for word, count in all_counts.items():
        if count >= min_count:
            rv = beta(a=count, b=N-count)
            freq = count / N
            if word in fcounts:
                count_f = fcounts[word]
                freq_f = count_f / f_N
                if freq < freq_f:  # more frequent in female than in overall
                    p = rv.sf(freq_f)
                    f_associated.append((word, p, count, count_f))
            if word in mcounts:
                count_m = mcounts[word]
                freq_m = count_m / m_N
                if freq < freq_m:  # more frequent in male than in overall
                    p = rv.sf(freq_m)
                    m_associated.append((word, p, count, count_m))
    print('Num female-associated:', len(f_associated))
    print('Num male-associated:', len(m_associated))

    f_associated = sorted(f_associated, key=lambda x:x[1])
    m_associated = sorted(m_associated, key=lambda x:x[1])
    return f_associated, m_associated

def filter_associations_on_lemma_and_pos(ass, valid_pos=None, invalid_pos=None, blacklist=STOPWORDS):
    filtered = []
    for tuple in ass:
        lemma, pos = tuple[0]
        if valid_pos is None or pos in valid_pos:
            if invalid_pos is None or pos not in invalid_pos:
                if is_valid_lemma(lemma, blacklist):
                    filtered.append(tuple)
    return filtered

def is_valid_lemma(lemma, blacklist):
    return lemma.isalpha() and lemma not in blacklist and len(lemma) > 2 and len(lemma) < 20 and lemma.lower() == lemma

def filter_associations_on_p(associations, p_thresh, print_filtered=False):
    filtered = []
    for tuple in associations:  # already ordered by largest to smallest cdf
        if tuple[1] <= p_thresh:
            filtered.append(tuple)
        else:
            break
    print('Number of associations with p <= {}: {}'.format(round(p_thresh, 3), len(filtered)))
    if print_filtered:
        for i, tuple in enumerate(filtered):
            print('{}. {}, p={}'.format(i+1, tuple[0], tuple[1]))
    return filtered

def print_top_n(f_ass, m_ass, top_n=25):
    print('Most Female')
    for i, (word, p, _, _) in enumerate(f_ass[:top_n]):
        print('{}. {}, p={}'.format(i+1, word, round(p, 4)))
    print('\nMost Male')
    for i, (word, p, _, _) in enumerate(m_ass[:top_n]):
        print('{}. {}, p={}'.format(i+1, word, round(p, 4)))

def print_top_n_per_pos(f_ass, m_ass, top_n=25):
    for pos in ['NOUN', 'VERB', 'ADJ']:
        print('\nFiltering on only lemmas with pos={}...'.format(pos))
        f_ass_within_pos = filter_associations_on_lemma_and_pos(f_ass, valid_pos={pos})
        m_ass_within_pos = filter_associations_on_lemma_and_pos(m_ass, valid_pos={pos})
        print('{} female, {} male'.format(len(f_ass_within_pos), len(m_ass_within_pos)))
        print('Most Female')
        for i, (word, p, _, _) in enumerate(f_ass_within_pos[:top_n]):
            print('{}. {}, p={}'.format(i+1, word, round(p, 4)))
        print('\nMost Male')
        for i, (word, p, _, _) in enumerate(m_ass_within_pos[:top_n]):
            print('{}. {}, p={}'.format(i+1, word, round(p, 4)))

if __name__ == '__main__':
    f_counts, m_counts = get_tok_counts_from_balanced_prof_corpus()
    f_ass, m_ass = beta_scoring_from_counts(f_counts, m_counts)
    pickle.dump((f_ass, m_ass), open(PATH_TO_PROF_PROCESSED + 'lex.pkl', 'wb'))

    # f_ass, m_ass = pickle.load(open(PATH_TO_PROF_PROCESSED + 'lex.pkl', 'rb'))
    f_ass = filter_associations_on_lemma_and_pos(f_ass, valid_pos={'NOUN', 'VERB', 'ADJ'})
    print('Num female words:', len(f_ass))
    m_ass = filter_associations_on_lemma_and_pos(m_ass, valid_pos={'NOUN', 'VERB', 'ADJ'})
    print('Num male words:', len(m_ass))

    alpha = 0.05
    alpha /= len(f_ass) + len(m_ass)  # Bonferroni correction
    print('Adjusted alpha:', round(alpha, 10))

    sig_f_ass = filter_associations_on_p(f_ass, p_thresh=alpha)
    sig_m_ass = filter_associations_on_p(m_ass, p_thresh=alpha)
    print('Total number of sig words:', len(sig_f_ass) + len(sig_m_ass))
    print_top_n_per_pos(sig_f_ass, sig_m_ass, top_n=100)