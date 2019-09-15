import rltk
import re
from utils import tokenize, manufacturers_list

class BaseRecord(rltk.Record):
    """
    Base record class for both datasets
    """
    def __init__(self, raw_object):
        super().__init__(raw_object)
        self.YEAR_MAX = 2050
        self.YEAR_MIN = 1970

    @rltk.cached_property
    def id(self):
        return self.raw_object['id']
    
    @rltk.cached_property
    def name(self):
        return self.raw_object['title']

    @rltk.cached_property
    def name_tokenized(self):
        tokens = tokenize(self.raw_object['title'])
        return tokens
    
    @rltk.cached_property
    def description_tokenized(self):
        if self.raw_object['description'] == '':
            return None
        tokens = tokenize(self.raw_object['description'])
        if len(tokens) == 0:
            return None
        return tokens
    
    @rltk.cached_property
    def version(self):
        version_pattern = re.compile(r'\d+\.\d+')
        matches = version_pattern.findall(self.name)
        if len(matches) > 1:
            print('name: {} id: {} more than one version detected'.format(self.name, self.id))
            return matches[0]
        elif len(matches) == 1:
            return matches[0]
        else:
            return None
    
    @rltk.cached_property
    def year(self):
        year_pattern = re.compile(r'\d{4}')
        matches = year_pattern.findall(self.name)

        if len(matches) <= 0:
            return None

        year = int(matches[0])

        if len(matches) > 1:
            print('name: {} id: {} more than one year detected'.format(self.name, self.id))
        
        if year >= self.YEAR_MAX or year <= self.YEAR_MIN:
            return None
        
        return year
    