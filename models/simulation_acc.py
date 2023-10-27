from dateutil import parser
import numpy as np
class simulation_acc:
    '''
    "Class to simulate the performance of a financial account. 
    Keeping track of the starting balance, end date, and the Net Asset Value (NAV) which is calculated as the sum of the balance and unrealized profits/losses."
    '''
    def __init__(self, start, end, sim_balance):
        self.start = parser.parse(start) if isinstance(start, str) else start
        self.end = parser.parse(end) if isinstance(end, str) else end
        self.starting_balance = sim_balance
        self.sim_balance = sim_balance
        self.sim_unrealized = 0
        self.sim_realized = 0
        self.sim_realized_perc = 0
        self.sim_NAV = self.sim_balance + self.sim_unrealized
        self.sim_trade_count = 0
        self.sim_long = {}
        self.sim_short = {}
        self.sim_long_trades = []
        self.sim_short_trades = []
        self.sim_margin_used = 0
        self.sim_margin_available = 0
        self.leverage = 50
        self.max_drawdown = 0
        self.max_equity = sim_balance
        self.margin_closeout_perc = 0
        self.volatility_score = []
        
    def calculate_margin(self):
        self.sim_margin_available = self.sim_balance - self.sim_margin_used + self.sim_unrealized
        self.margin_closeout_perc = (self.sim_margin_used / self.sim_NAV)

        return self.sim_margin_available


    def calculate_max_drawdown(self):
        drawdown = (self.max_equity - self.sim_NAV) / self.max_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        if self.sim_NAV > self.max_equity:
            self.max_equity = self.sim_NAV

        return self.max_drawdown