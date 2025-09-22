from lightlemma import lemmatize, tokenize

class CustomLemmaTokenizer(object):
    """Custom english tokenizer and lemmatizer
    For other languages, there is either no lemmatization (italian: tavoli -> tavoli)
    or limited lemmatization (french: marquantes -> marquante)"""
    def __init__(self) -> None :
        # Memory is used to revert the lemmatization for readability
        self.__memory = {} 

    def __lemmatize_and_memorise(self, token) -> str:
        lemma = lemmatize(token)
        if token not in self.__memory:
            self.__memory[lemma] = token
        return lemma

    def __call__(self, text) -> list[str]: 
        return [self.__lemmatize_and_memorise(token) for token in tokenize(text)]
    
    def __remember(self, lemma) -> str: 
        try : 
            return self.__memory[lemma]
        except:
            return lemma

    def revert(self, lemmas : list[str]):
        return [self.__remember(lemma) for lemma in lemmas]