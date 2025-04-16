import re
import string
import emoji
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()

STOPWORDS = {
    "yoruba": set("""
        tó àwọn ń pé tí ṣe máa ní wọ́n ó ni kí sì sí jẹ́ ti bá a fi lè náà kan fún
        láti rẹ̀ sọwọn ohun àti bí ninu wà kò wa wá yìí ló rí kó ká lọ o mi mo
        gbogbo ọmọ ò í ẹ á ọrọ ẹni ara fẹ́ i wo gbe pa ọdun di kì yóò
    """.split()),

    "igbo": set("""
        na onye gi ka ya m ndi a o nke di ihe i no bu anyi ga igbo gị ike chukwu ọ
        biko e ma nwanne oma aka mma nna nwa ndị ha egwu the unu kwa ebe nne eme
        isi bụ dị maka nwoke si anyị okwu ji ego obi anya otu mgbe oge ị onwe ala mana eji
    """.split()),

    "hausa": set("""
        da allah ya a ba ta na wannan ne kuma su sai mu yan ko mai dai shi ka yayi to
        amma ga haka yi daga ma duk kai masu wani sun zai sa ina nan idan wanda an ce ke
        suka cikin wa daya kasa iya yake domin in kan yana aka za me don har ake gaba akwai yadda abin
    """.split()),

    "pidgin": set("""
        di abeg wetin sef abi wahala comot shey pikin pesin sey deh weda pipo kon tok dis im
        bin wit fit de oda don e dat wen d kain tins na wia dey
    """.split())
}


def preprocess_text(text, language='yoruba'):
    """
    Preprocess text for the given language by:
    - Lowercasing
    - Demojizing
    - Removing URLs and mentions
    - Removing punctuation
    - Tokenizing and removing stopwords
    - Lemmatizing
    """
    if language.lower() not in STOPWORDS:
        raise ValueError(f"Unsupported language '{language}'. Choose from {list(STOPWORDS.keys())}.")

    stopwords = STOPWORDS[language.lower()]
    
    text = text.lower()
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)
