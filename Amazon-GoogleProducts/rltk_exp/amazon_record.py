import rltk
import re
from utils import tokenize
from base_record import BaseRecord

class AmazonRecord(BaseRecord):
    def __init__(self, raw_object):
        super().__init__(raw_object)

    @rltk.cached_property
    def manufacturer(self):
        return self.raw_object['manufacturer']

    @rltk.cached_property
    def price(self):
        price = float(self.raw_object['price'])
        if price == 0:
            return None
        return price