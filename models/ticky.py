import datetime


class ticky(object):
    """
    holds adding size and stop loss ticket data for each individual trading pair
    returns itself
    """
    
    def __init__(self, symbol, adds=0, add_price=0, original=0, BUY=True, SELL=False, candles=None, s_r_candles=None, phone=False):
        self.symbol = symbol
        self.adds = adds
        self.add_price = add_price
        self.original = original
        self.BUY = BUY
        self.SELL = SELL
        self.size = []
        self.candles = candles
        self.s_r_candles = s_r_candles
        self.phone = phone
        self.add_price_increment = 0.0025
        self.prompted = False