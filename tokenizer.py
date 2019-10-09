def try_or_retry(f, handle):
    try:
        ret = f()
    except:
        handle()
        ret = f()
    return ret


def get_nltk_tokenizer():
    import nltk
    def import_tokenizer():
        global word_tokenize
        from nltk.tokenize import word_tokenize
    def handle():
        nltk.download('punkt')
    try_or_retry(import_tokenizer, handle)

    contractions = ["'d", "'ll", "'m", "'re", "'s", "'t", "'ve"]

    def tokenize(text):
        space = [False] * (len(text) + 1)
        delete = [False] * len(text)
        i = 0
        while i < len(text):
            if text[i] in ".*-',!?~":
                j = i
                while j < len(text) and text[j] == text[i]:
                    j += 1
                if j - i >= 2 or text[i] in "*~" or (text[i] in "-" and (i == 0 or text[i-1] == ' ' or j == len(text) or text[j] == ' ')):
                    space[i] = True
                    space[j] = True
                if text[i] in ".!-" and j - i > 3:
                    for k in range(i + 3, j):
                        delete[k] = True
                i = j
            else:
                i += 1
        for i in range(1, len(text)-1):
            if text[i-1:i+1] == " '":
                j = i + 1
                while j < len(text) and text[j].isalnum():
                    j += 1
                if text[i:j].lower() not in contractions:
                    space[i+1] = True
        for i in range(1, len(text)-1):
            if text[i] == '/' and text[i-1] != '/' and text[i+1] != '/':
                space[i] = True
                space[i+1] = True
        new_text = []
        for i in range(len(text)):
            if i > 0 and space[i]:
                new_text.append(' ')
            if not delete[i]:
                new_text.append(text[i])
        text = ''.join(new_text)

        tokens = word_tokenize(text)

        new_tokens = []
        for token in tokens:
            new_token = [token]
            if token == 'i':
                new_token = ['I']
            elif token[0] == "'" and len(token) > 1:
                if token[-1] == "'":
                    if len(token) > 2:
                        new_token = [token[0], token[1:-1], token[-1]]
                elif token.lower() not in contractions:
                    new_token = [token[0], token[1:]]
            elif len(token) > 1 and token[0] == "-" and token[1] != "-":
                if token[-1] == "-":
                    if len(token) > 2:
                        new_token = [token[0], token[1:-1], token[-1]]
                else:
                    new_token = [token[0], token[1:]]
            new_tokens.extend(new_token)
        tokens = new_tokens

        return tokens

    return tokenize


def get_stanfordnlp_tokenizer():
    import stanfordnlp
    def get_nlp():
        global nlp
        nlp = stanfordnlp.Pipeline(processors='tokenize')
    def handle():
        stanfordnlp.download('en')
    try_or_retry(get_nlp, handle)

    def tokenize(text):
        doc = nlp(text)
        tokens = []
        for sentence in doc.sentences:
            tokens.extend(word.text for word in sentence.words)
        return tokens

    return tokenize


def get_transformers_tokenizer():
    import transformers
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
    return tokenizer.tokenize


def get_tokenizer(tokenizer_name):
    return {
        'nltk': get_nltk_tokenizer,
        'stanfordnlp': get_stanfordnlp_tokenizer,
        'transformers': get_transformers_tokenizer,
    }[tokenizer_name]()
