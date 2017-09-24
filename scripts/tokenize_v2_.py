# -*- coding: utf-8 -*-

import re
import tensorflow as tf

class Tokenizer:

    def __init__(self, path):

        self.abbrevs = []
        abbrev = tf.gfile.GFile(path + '/abbrev.ru.txt')

        for line in abbrev:
            line = line.strip()
            self.abbrevs.append(line)
        abbrev.close()

        # create the arr with words from closed class

        self.closed_class = []
        cl_class = tf.gfile.GFile(path + '/new_closed_class.txt')

        for line in cl_class:
            line = line.strip()
            self.closed_class.append(line)
        cl_class.close()

        # add defis abbreviations to arr with words from closed class

        defis_abbr = tf.gfile.GFile(path + '/defis_abbreviations.ru.txt')
        for line in defis_abbr:
            line = line.strip()
            self.closed_class.append(line)
        defis_abbr.close()

        # create the arr with toponim parts

        self.topon_parts = []
        t_parts = tf.gfile.GFile(path +  '/topon_parts.txt')
            # codecs.open(FLAGS.tok_path + 'topon_parts.txt', 'r', 'utf-8')
        for line in t_parts:
            line = line.strip()
            self.topon_parts.append(line)
        t_parts.close()

        # open a file


        self.url_1 = re.compile(u'(?:https?|ftp|www\\.)[a-z0-9/\\._:]+') ### here go urls of common type: file, https, etc
        self.url_2 = re.compile(u'((?:[a-zа-я]+\\.)+(?:biz|com|edu|gov|info|int|mil|name|net|org|pro|travel|xxx|ac|ad|'
                           u'ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|'
                           u'br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|'
                           u'dj|dk|dm|do|dz|ec|ee|eg|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|'
                           u'gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|sv|il|im|in|io|iq|ir|is|it|je|jm|'
                           u'jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|'
                           u'mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|'
                           u'nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|sv|rw|sa|sb|sc|sd|'
                           u'se|sg|sh|si|sj|sk|sl|sm|sn|so|sr|st|su|sv|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|'
                           u'tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw|рф|ру))') # tokens that resemble to url
        self.email = re.compile(u'([_a-z0-9-]+(?:\\.[_a-z0-9-]+)*@(?:[a-zа-я]+\\.)+(?:biz|com|edu|gov|info|int|mil|name|'
                           u'net|org|pro|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|'
                           u'be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|'
                           u'cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|'
                           u'ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|sv|il|im|in|io|'
                           u'iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|'
                           u'ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|'
                           u'nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|sv|rw|sa|'
                           u'sb|sc|sd|se|sg|sh|si|sj|sk|sl|sm|sn|so|sr|st|su|sv|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|'
                           u'to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw|рф|ру))') # tokens that resemble to e-mail
        self.smile_1 = re.compile(u'([:=]\-?\)+|[:=]\-?\(+)') ### tokens that resemble to smile
        self.smile_2 = re.compile(u'[ツ‿♡ノヽ・∀*)(ω_/¯^OoОо\\\]{3,}') ###
        self.number_1 = re.compile(u'([12][0-9]{3}-(?:о?м|о?го|о?е|ы?х)|'
                              u'[0-9]+[.,][0-9]+|\d+-\d+|\d+-(?:о?м|о?го|о?е|ы?х)|\d+/\d+)') ### cases 2008-th and 1.5,1.6, 15-16
        self.number_2 = re.compile(u'(?:^| )((?:[A-Za-z]?[0-9]+[A-Za-z]?)|(?:[+-]?\d+(?:[.,]\d+(?:e-?\d+)?)?)|'
                              u'(?:\\\[0-3]{1,2})|(?:0x[0-9a-fA-F]{1,16}))') # natural & floating point numbers | octets | hex digits
        self.acronym = re.compile(u'(?:[A-ZА-ЯЁ]\\. ?){1,10}[A-ZА-ЯЁ]\\.') ### tokens that resemble to acronym
        self.rep_punct = re.compile(u'[-?!.]{2,}') ### repeated punctuation
        self.hashtag = re.compile(u'\#[^-“”«»,.?!:;)(\\]\\[`\"„†‡‹}{\'%…‰‘’•–—›\\\|\r\n ]*') ### comments
        self.sgml = re.compile(u'^<.*>$') ### sgml tags
        self.defis = re.compile(u'[^-“”«»,.?!:;)(\\]\\[`\"„†‡‹}{\'%…‰‘’•–—›\\\/|\r\n ]+?(?:-[^-“”«»,.?!:;)'
                           u'(\\]\\[`\"„†‡‹}{\'%…‰‘’•–—›\\\|\r\n ]+)+') #hyphen
        self.short_1 = re.compile(u'[^-“”«»,.?!:;)(\\]\\[`\"„†‡‹}{\'%…‰‘’•–—›\\\/|\r\n ]+?\\.')
        self.short_2 = re.compile(u'[^-“”«»,.?!:;)(\\]\\[`\"„†‡‹}{\'%…‰‘’•–—›\\\/|\r\n0-9 ]{2}/[^-“”«»,.?!:;)(\\]\\[`\"„†‡‹}{\'%…‰‘’•–—›\\\/|\r\n ]')
        self.date = re.compile(u'(?:[1-9][0-9]{3}|(?:0[1-9]|[12][0-9]|3[01]))[\./,]'
                          u'(?:[01][0-9])[\\./,](?:[1-9][0-9]{3}|0[1-9]|[12][0-9]|3[01])[\.\/,]?')

    def tokenize(self,text):

        # reg exp compilation


        # change

        text = re.sub('(\\)+)(\\()', '\\1 \\2', text) ## for smiles and brackets ')))(lflf)'

        smiles_type1 = self.smile_1.findall(text)
        text = self.smile_1.sub(' SMILETOKEN_TYPE_109484712 ', text)

        smiles_type2 = self.smile_2.findall(text)
        text = self.smile_2.sub(' SMILETOKEN_TYPE_209484712 ', text)

        urls_type1 = self.url_1.findall(text)
        text = self.url_1.sub(' URLTOKEN_TYPE_109484712 ', text)

        urls_type2 = self.url_2.findall(text)
        text = self.url_2.sub(' URLTOKEN_TYPE_209484712 ', text)

        emails = self.email.findall(text)
        text = self.email.sub(' EMAILTOKEN_TYPE_109484712 ', text)

        dates = self.date.findall(text)
        text = self.date.sub(' DATETOKEN_TYPE_109484712 ', text)

        numbers_type1 = self.number_1.findall(text)
        text = self.number_1.sub(' NUMBERTOKEN_TYPE_109484712 ', text)

        numbers_type2 = self.number_2.findall(text)
        text = self.number_2.sub(' NUMBERTOKEN_TYPE_209484712 ', text)

        defises = self.defis.findall(text)
        text = self.defis.sub(' DEFISTOKEN_TYPE_109484712 ', text)

        acronyms = self.acronym.findall(text)
        text = self.acronym.sub(' ACRONYMTOKEN_TYPE_109484712 ', text)

        rep_puncts = self.rep_punct.findall(text)
        text = self.rep_punct.sub(' REPPUNCTTOKEN_TYPE_109484712 ', text)

        hashtags = self.hashtag.findall(text)
        text = self.hashtag.sub(' HASHTAGTOKEN_TYPE_109484712 ', text)

        sgmls = self.sgml.findall(text)
        text = self.sgml.sub(' SGMLTAGTOKEN_TYPE_109484712 ', text)

        shorts_type2 = self.short_2.findall(text)
        text = self.short_2.sub(' SHORTENINGTOKEN_TYPE_209484712 ', text)

        shorts_type1 = self.short_1.findall(text)
        text = self.short_1.sub(' SHORTENINGTOKEN_TYPE_109484712 ', text)

        # normalization

        text = text.strip()
        text = re.sub(u' ', ' ', text) #replace non-break spaces
        text = re.sub('[\n\t]', ' ', text) # replace newlines and tab characters with blanks
        text = re.sub(' ', ' ', text)
        text = re.sub(u'([-“”«»,.?!:;)(\\]\\[`"„†‡‹}{\'%…‰‘’•–—›\\\|/])', ' \\1 ', text)
        text = re.sub(u'…', ' ... ', text)
        text = re.sub(' +', ' ', text)

        # return

        em = 0
        url1 = 0
        url2 = 0
        sm1 = 0
        sm2 = 0
        num1 = 0
        num2 = 0
        acr = 0
        rp = 0
        hsh = 0
        sgm = 0
        df = 0
        sh1 = 0
        sh2 = 0
        dt = 0

        # w = codecs.open('output.txt', 'w', 'utf-8')

        text_tokenized = text.split()
        #print text_tokenized

        result = []

        for token in text_tokenized:
            rest = ''
            if token == 'EMAILTOKEN_TYPE_109484712':
                result.append(emails[em])
                em += 1
            elif token == 'URLTOKEN_TYPE_109484712':
                result.append(urls_type1[url1])
                url1 += 1
            elif token == 'URLTOKEN_TYPE_209484712':
                 result.append(urls_type2[url2])
                 url2 += 1
            elif token == 'SMILETOKEN_TYPE_109484712':
                result.append(smiles_type1[sm1])
                sm1 += 1
            elif token == 'SMILETOKEN_TYPE_209484712':
                result.append(smiles_type2[sm2])
                sm2 += 1
            elif token == 'NUMBERTOKEN_TYPE_109484712':
                result.append(numbers_type1[num1])
                num1 += 1
            elif token == 'NUMBERTOKEN_TYPE_209484712':
                result.append(numbers_type2[num2])
                num2 += 1
            elif token == 'ACRONYMTOKEN_TYPE_109484712':
                result.append(acronyms[acr])
                acr += 1
            elif token == 'REPPUNCTTOKEN_TYPE_109484712':
                result.append(rep_puncts[rp])
                rp += 1
            elif token == 'HASHTAGTOKEN_TYPE_109484712':
                result.append(hashtags[hsh])
                hsh += 1
            elif token == 'SGMLTAGTOKEN_TYPE_109484712':
                result.append(sgmls[sgm])
                sgm += 1
            elif token == 'DATETOKEN_TYPE_109484712':
                result.append(dates[dt])
                dt += 1
            elif token == 'SHORTENINGTOKEN_TYPE_209484712':
                # print shorts_type2[sh2]
                result.append(shorts_type2[sh2])
                sh2 += 1
            elif token == 'SHORTENINGTOKEN_TYPE_109484712':
                if shorts_type1[sh1] in self.abbrevs:
                    result.append(shorts_type1[sh1])
                else:
                    result.append(shorts_type1[sh1][:-1])
                sh1 += 1
            elif token == 'DEFISTOKEN_TYPE_109484712':
                def_parts = defises[df].split('-')
                if re.search(u'^(то|де|ка|таки)$', def_parts[-1].lower()):
                    if defises[df] not in self.closed_class:
                        defises[df] = '-'.join(def_parts[:-1]) # token
                        rest = '-\n' + def_parts[-1] # cut particle
                def_parts = defises[df].split('-')
                def check_func(word, closed_class = self.closed_class, topon_parts = self.topon_parts):
                    spl_word = word.split('-')
                    if len(spl_word) == 1:
                        return word, 0
                    elif word.lower() in closed_class:
                        return word, 0 # found in file (whole)
                    elif spl_word[0].lower() + '-' in closed_class:
                        return word, 0 # found in file (part)
                    elif '-' + spl_word[1].lower() in closed_class:
                        return word, 0 # found in file (part)
                    elif '-' + spl_word[1].lower() + '-' in topon_parts:
                        return word, 0 # found in toponim parts
                    elif re.search(u'[А-ЯЁ][а-яё]+', spl_word[0]) and re.search(u'[А-ЯЁ][а-яё]+', spl_word[-1]): # last\first part has the first letter capital and other lower
                        return word, 0
                    return ['<o composite="true">\n', '\n-\n'.join(spl_word), '\n</o>'], 1

                if len(def_parts) <= 3:
                    check, comp = check_func(defises[df])
                    result.append(''.join(check))

                elif len(def_parts) > 3:
                    part_1 = def_parts[:len(def_parts)/2]
                    part_2 = def_parts[len(def_parts)/2:]
                    check_1, comp_1 = check_func('-'.join(part_1))
                    check_2, comp_2 = check_func('-'.join(part_2))
                    if comp_1 == 1:
                        check_1.pop(0)
                        check_1.pop(-1)
                        check_1 = ''.join(check_1)
                    if comp_2 == 1:
                        check_2.pop(0)
                        check_2.pop(-1)
                        check_2 = ''.join(check_2)
                    result.append('<o composite="true">\n' + check_1 + '\n-\n' + check_2)
                if rest:
                    result.append(rest)
                df += 1
            else:
                result.append(token)

        return result
