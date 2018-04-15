def words(lang='english'):
    if lang == 'english':
        f = open('pythia/stopwords/stopwords_en.txt', 'r')
        stop_list = [word.replace('\n', '') for word in f.readlines()]
    else:
        raise NotImplementedError
    return stop_list
