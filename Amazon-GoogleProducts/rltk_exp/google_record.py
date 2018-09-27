import rltk
import re
from utils import tokenize, manufacturers_list, aliases
from base_record import BaseRecord

class GoogleRecord(BaseRecord):
    """
    Record class for google records
    """
    def __init__(self, raw_object):
        super().__init__(raw_object)
    
    @rltk.cached_property
    def name(self):
        return self.raw_object['name']

    @rltk.cached_property
    def name_tokenized(self):
        tokens = tokenize(self.raw_object['name'])
        return tokens
    
    @rltk.cached_property
    def manufacturer(self):
        if self.raw_object['manufacturer'] != '':
            return self.raw_object['manufacturer']
            
        tokens = self.name_tokenized

        for word_len in range(min(5, len(tokens)), 0, -1):
            i = 0; j = i + word_len
            while j <= len(tokens):
                name = ' '.join(tokens[i:j])
                if name in manufacturers_list:
                    try:
                        name = aliases[name]
                    except KeyError:
                        pass
                    return name
                i += 1; j += 1
    
        return ''

    @rltk.cached_property
    def price(self):
        patt = re.compile(r'\d+\.?\d{0,2}')
        price = patt.findall(self.raw_object['price'])
        if len(price) > 0:
            price = float(sorted(price)[-1])
        else:
            price = None
            print("Null Price __ id: {}, price: {}".format(self.id, self.raw_object['price']))
        return price