from googletrans import Translator
translator = Translator()
text = 'कंप्यूटर पर हिंदी में टायपिंग करना बहुत आसान बना दिया है'
srcLang = 'hi'
destLang   = 'en'
print(translator.translate('कंप्यूटर पर हिंदी में टायपिंग करना बहुत आसान बना दिया है', dest=destLang , src=srcLang).text)

