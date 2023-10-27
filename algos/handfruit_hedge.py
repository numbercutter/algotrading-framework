
import time
from tools import time_tools
from tools.init_tools import init_tools
from tools import size_tools
import schedule
import datetime
import pandas as pd
from exploration.plotting import CandlePlot
from models import ticky
from models import simulation_acc
from technicals.indicators import Indicators
from api.oanda_api_sim import OandaApiSim
import constants.defs as defs
import plotly.graph_objs as go
from plotly.subplots import make_subplots
class handfruit_hedge:

    ERROR_LOG = "handfruit_hedge_error"
    MAIN_LOG = "handfruit_hedge_main"
    SLEEP = 300
    SLEEP_SIM = 0.3

    def __init__(self, symbol="EUR_USD", account_id=None, api_k=None, phone_n=None, log_m=None, simulation=False, BUY=False, SELL=False):
        # Initialize instance variables
        self.account_id = account_id
        self.symbol = symbol
        self.api_k = api_k
        self.phone_n = phone_n
        self.log_m = log_m
        self.simulation = simulation
        self.BUY = BUY
        self.SELL = SELL

        
        # Set up log names
        self.error_log = f'{self.log_m}_{handfruit_hedge.ERROR_LOG}'
        self.main_log = f'{self.log_m}_{handfruit_hedge.MAIN_LOG}'
        print(f'error log: {self.error_log}')
        print(f'main log: {self.main_log}')
        
        # Initialize AlgorithmTools instance
        self.algorithm_tools = init_tools(
            symbol=symbol,
            simulation_data=simulation_acc,
            account_id=account_id,
            error_log=self.error_log,
            main_log=self.main_log,
            phone_n=phone_n
        )
        
        # Set up log files
        self.algorithm_tools.setup_logs()
        

        # Initialize logging
        self.simulation_data = dict()
        self.simulation_results = dict()


        # Connect to API based on simulation flag
        if not self.simulation:
            self.api = OandaApiSim(
                api_key=self.api_k,
                oanda_url=defs.OANDA_URL,
                account_id=self.account_id
            )
        else:
            self.api = OandaApiSim(
                api_key=self.api_k,
                oanda_url=defs.OANDA_URL,
                account_id=self.account_id
            )

            self.simulation_data[self.symbol] = simulation_acc.simulation_acc("2015-01-01T00:00:00Z", "2016-01-01T00:00:00Z", 10000)

        # Log the start of the bot
        self.algorithm_tools.log_to_main("SR Bot started")
        self.algorithm_tools.log_to_error("SR Bot started")

        # Initialize symbolData and Indicators dictionaries
        self.symbolData = dict()
        self.Indicators = dict()
        self.Indicators[self.symbol] = Indicators(self.symbol)
        self.volatility_scores = []
        self.realized_gains = []
        self.snapshot = dict()

        # Load and set up simulation data if simulation flag is True
        if self.simulation:
            sim = self.simulation_data[self.symbol]
            raw_candles_df_m1, raw_candles_df_m5, raw_candles_df_m15, raw_candles_df_h1, raw_candles_df_h4, raw_candles_df_d = self.algorithm_tools.load_data(sim, pair=symbol)
            candles_copy = {
                "M1": raw_candles_df_m1.copy().reset_index(drop=True),
                "M5": raw_candles_df_m5.copy().reset_index(drop=True),
                "M15": raw_candles_df_m15.copy().reset_index(drop=True),
                "H1": raw_candles_df_h1.copy().reset_index(drop=True),
                "H4": raw_candles_df_h4.copy().reset_index(drop=True),
                "D": raw_candles_df_d.copy().reset_index(drop=True)
            }
            self.algorithm_tools.log_to_main(candles_copy)
            self.symbolData[self.symbol] = ticky.ticky(self.symbol, candles=self.collect_trend_indicators(candles_copy), s_r_candles=self.collect_sr_indicators(candles_copy))
            
             # Pass the simulation candles to collect indicators
        else:
            self.symbolData[self.symbol] = ticky.ticky(self.symbol)
            self.collect_candles(['M5', 'M15', 'D']) # Otherwise, collect candles from the API as usual.

    def collect_trend_indicators(self, candles_dict: dict):
        '''
        Collect and calculate trend indicators (RSI and EMA) for each granularity based on the candles data. 
        The resulting indicators are stored in the ind attribute of the si object and logged to the main log.
        '''
        si = self.Indicators[self.symbol]

        for granularity, candles in candles_dict.items():
            r = Indicators.RSI(candles)
            e_50 = Indicators.EMA(candles, 50)
            e_9 = Indicators.EMA(candles, 9)

            candles['RSI_14'] = r["RSI_14"]
            candles['EMA_50'] = e_50["EMA_50"]
            candles['EMA_9'] = e_9["EMA_9"]

            si.ind[f'RSI_14_{granularity}'] = r["RSI_14"].iloc[-1]
            si.ind[f'EMA_50_{granularity}'] = e_50["EMA_50"].iloc[-1]
            si.ind[f'EMA_9_{granularity}'] = e_9["EMA_9"].iloc[-1]

            #self.algorithm_tools.log_to_main(f'INDICATORS: {si.ind} TIME: {datetime.datetime.now()}')

        return candles_dict
    
    def collect_sr_indicators(self, candles_dict: dict):
        '''
        Collect and calculate support/resistance indicators (mm levels) for each granularity based on the candles data. 
        The resulting indicators are stored in the ind attribute of the si object and logged to the main log.
        '''
        si = self.Indicators[self.symbol]

        sup_res_values = {
            'S5': (40, 0.0002, 5000),
            'M1': (60, 0.0005, 5000),
            'M5': (40, 0.0005, 2000),
            'M15': (20, 0.0020, 1000), 
            'H1': (20, 0.0030, 500),
            'H4': (20, 0.0040, 500),
            'D': (20, 0.0060, 1000),
        }

        for granularity, candles in candles_dict.items():
            if granularity in ['M5', 'M15', 'D']:
                if not self.simulation:
                    s_r = Indicators.mm(candles, sup_res_values.get(granularity, (0, 0))[0], sup_res_values.get(granularity, (0, 0))[1])

                    si.ind[f'S_R_{granularity}'] = s_r

                if self.simulation:
                    sim = self.simulation_data[self.symbol]
                    s_r = pd.read_pickle(f"{'./data/oanda_2005_2023/'}_{'support_resistance'}_{self.symbol}_{granularity}.pkl")
                    #self.algorithm_tools.log_to_main(s_r)
                    s_r = s_r[(s_r.time>=sim.start)&(s_r.time<sim.end)]
                    s_r.reset_index(drop=True, inplace=True)
                    candles['support_resistance'] = s_r['support_resistance']

            #self.algorithm_tools.log_to_main(f'INDICATORS: {si.ind} TIME: {datetime.datetime.now()}')

        return candles_dict
    
    def collect_candles(self, granularities):
        '''
        collect_indicators function computes the RSI, EMA, and support-resistance values for the given candles and updates the Indicators object for the symbol.
        '''
        si = self.Indicators[self.symbol]
        
        candles_dict = {}
        count_dict = {
            'S5': 4000,
            'M1': 3000,
            'M5': 2000,
            'M15': 1000, 
            'H1': 750,
            'H4': 500,
            'D': 1000,
        }
        
        for granularity in granularities:
            count = count_dict.get(granularity, 100)
            raw_candles_df = self.api.get_candles_df(self.symbol, granularity=granularity, count=count)
            candles_copy = raw_candles_df.copy()
            candles_copy.reset_index(drop=True, inplace=True)
            candles_dict[granularity] = candles_copy

        self.collect_trend_indicators(candles_dict)
        self.collect_sr_indicators(candles_dict)
        
        for granularity, candles in candles_dict.items():
            plot = CandlePlot(candles)
            if si:
                plot.plot_img(lines=si.ind[f'S_R_{granularity}'], filename=f"support_resistance_{granularity}.png")

    def OnData(self):
        dates_to_skip = time_tools.news_dates
        # Fetching account, symbol and indicator data
        sd = self.symbolData[self.symbol]
        si = self.Indicators[self.symbol]

        if self.simulation:
            sim = self.simulation_data[self.symbol]


            # Create boolean masks to filter the rows of each dataframe
            mask_m15 = sd.candles['M15']['time'] == pd.Timestamp(self.current_timestamp).floor('15T')



            # Extract indicator values for the simulation current time using boolean masks
            RSI_14_D = sd.candles['D'][pd.Series(sd.candles['D']['time']).dt.floor('D') == pd.Timestamp(self.current_timestamp).floor('D')]['RSI_14'].values[0]          
            #RSI_14_H1 = sd.candles['H1'].loc[sd.candles['H1']['time'] == pd.Timestamp(self.current_timestamp).floor('H'), 'RSI_14'].values
            RSI_14_M15 = sd.candles['M15'].loc[mask_m15, 'RSI_14'].values[0]
            EMA_50 = sd.candles['M15'].loc[mask_m15, 'EMA_50'].values[0]
            EMA_9 = sd.candles['M15'].loc[mask_m15, 'EMA_9'].values[0]
            S_R_15 = sd.s_r_candles['M15'].loc[mask_m15, 'support_resistance'].values[0]


            #print((f'>> sd.candles @ : {sd.candles["D"]["time"]}'))
            #print((f'>> pd.timesta @ : {pd.Timestamp(self.current_timestamp).floor("D")}'))
            #print(sd.s_r_candles['D'])
            S_R_D = sd.s_r_candles['D'][pd.Series(sd.candles['D']['time']).dt.floor('D') == pd.Timestamp(self.current_timestamp).floor('D')]['support_resistance'].values[0]
            S_R_5 = sd.s_r_candles['M5'][sd.candles['M5']['time'] == pd.Timestamp(self.current_timestamp).floor('5T')]['support_resistance'].values[0]

            
            
            # Get account and market information
            a_pos = self.api.get_position_summary(self.account_id, acc=sim)
            a_sum = self.api.get_account_summary(self.account_id, acc=sim)
            a_ask = self.api.get_ask_bid([self.symbol], t=self.current_timestamp, candles_df=sd.candles['M1'], sim_bool=self.simulation)
            EQUITY = float(a_sum['NAV'])
            TRADE_COUNT = int(a_sum['openTradeCount'])
            CURRENT_ASK = float(a_ask)
            WEIGHTED_PRICE = self.api.calculate_weighted_average_price(acc=sim)
            UNREALIZED = self.api.calculate_unrealized_profit_loss(acc=sim, wp=WEIGHTED_PRICE, ask=CURRENT_ASK)
            REALIZED = sim.sim_realized
            REALIZED_PERC = sim.sim_realized_perc
            MARGIN_REMAINING = sim.calculate_margin()
            MARGIN_CLOSEOUT_PERCENT = sim.margin_closeout_perc
            DRAWDOWN = sim.calculate_max_drawdown()
            current_str = self.current_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
            current_time = datetime.datetime.strptime(current_str, "%Y-%m-%dT%H:%M:%SZ")
            today = current_time.date()
            self.realized_gains.append(REALIZED)
            if len(self.realized_gains) >= 1000:
                v_score = Indicators.calculate_volatility(self.realized_gains, 500)
                self.volatility_scores.append(v_score)
            if len(self.realized_gains) < 1000:
                v_score = 0.5
                self.volatility_scores.append(v_score)

            # Round down the current time to the nearest H4 time
            current_time_h4 = pd.Timestamp(self.current_timestamp).floor('4H')

            # Get the last 200 H4 candles relative to the rounded down current time
            losing_candles = sd.candles['H4'].loc[sd.candles['H4']['time'] < current_time_h4][-800:]

            #trend_candles = sd.candles['H1'].loc[sd.candles['H1']['time'] < current_time_h4][-168:]

            #trend = Indicators.trend_indicator(trend_candles)
            #print(trend)
            # Add losing_candles to simulation results
            self.simulation_results.setdefault(self.symbol, {
                'time': [],
                'losing_candles': [],
                'sim_NAV': [],
                'sim_realized_perc': [],
                'margin_closeout_perc': [],
                'max_drawdown': [],
                'CURRENT_ASK': [],
            })
            self.simulation_results[self.symbol]['time'].append(current_time)
            self.simulation_results[self.symbol]['losing_candles'] = losing_candles
            self.simulation_results[self.symbol]['sim_NAV'].append(sim.sim_NAV)
            self.simulation_results[self.symbol]['sim_realized_perc'].append(sim.sim_realized_perc)
            self.simulation_results[self.symbol]['margin_closeout_perc'].append(sim.margin_closeout_perc)
            self.simulation_results[self.symbol]['max_drawdown'].append(sim.max_drawdown)
            self.simulation_results[self.symbol]['CURRENT_ASK'].append(CURRENT_ASK)

            if today in [date.date() for date in dates_to_skip]:
                print('skip')
                if TRADE_COUNT > 0:
                    LONG = a_pos['long']
                    SHORT = a_pos['short']
                    units_l = int(LONG['units'])
                    units_s = int(SHORT['units'])
                    if UNREALIZED > 0:
                        self.algorithm_tools.log_to_main(f'UNREALIZED PROFIT: {UNREALIZED}')
                        self.algorithm_tools.log_to_main(f'CLOSE $: {CURRENT_ASK}')
                        self.api.close_position(self.symbol, long_units=units_l, acc=sim, unrealized=UNREALIZED, ask=CURRENT_ASK, current_time=current_time)
                        return
                return
        if not self.simulation:
            sim = None
            RSI_14_M15 = float(si.ind['RSI_14_M15'])
            RSI_14_D = float(si.ind['RSI_14_D'])

            EMA_50 = float(si.ind['EMA_50_M15'])
            EMA_9 = float(si.ind['EMA_9_M15'])
            S_R_D = si.ind['S_R_D']
            S_R_15 = si.ind['S_R_M15']
            S_R_5 = si.ind['S_R_M5']
            a_pos = self.api.get_position_summary(self.account_id)
            a_sum = self.api.get_account_summary(self.account_id)
            a_ask, a_bid = self.api.get_ask_bid([self.symbol])
            UNREALIZED = float(a_sum['unrealizedPL'])
            REALIZED = float(a_sum['pl'])
            EQUITY = float(a_sum['NAV'])
            TRADE_COUNT = int(a_sum['openTradeCount'])
            CURRENT_ASK = float(a_ask)
            REALIZED_PERC = 0
            MARGIN_REMAINING = 0
            DRAWDOWN = 0
            MARGIN_CLOSEOUT_PERCENT = float(a_sum['marginCloseoutPercent'])
            today = datetime.datetime.now().date()
            current_time = self.current_timestamp
            if today in [date.date() for date in dates_to_skip]:
                if TRADE_COUNT > 0:
                    LONG = a_pos['long']
                    SHORT = a_pos['short']
                    units_l = int(LONG['units'])
                    units_s = int(SHORT['units'])
                    if UNREALIZED > 0:
                        self.algorithm_tools.log_to_main(f'UNREALIZED PROFIT: {UNREALIZED}')
                        self.algorithm_tools.log_to_main(f'CLOSE $: {CURRENT_ASK}')
                        self.api.close_position(self.symbol, long_units=units_l, acc=sim, unrealized=UNREALIZED, ask=CURRENT_ASK, current_time=current_time)
                        return
                return
            if not a_sum:
                return

        # Define position size multiplier
        pos_close_mult = 1

        # Define CLOSE list with original values
        CLOSE = [
            round(EQUITY * .00001, 2),
            round(EQUITY * .00011, 2),
            round(EQUITY * .00022, 2),
            round(EQUITY * .00033, 2),
            round(EQUITY * .00055, 2),
            round(EQUITY * (.00055 + (sd.original / 10000)), 2)
        ]

        # Modify CLOSE values with position size multiplier
        CLOSE = [round(value * pos_close_mult, 2) for value in CLOSE]
        



        top_half_D, bottom_half_D = Indicators.split_clusters(S_R_D)
        top_half_M15, bottom_half_M15 = Indicators.split_clusters(S_R_15)
        top_half_M5, bottom_half_M5 = Indicators.split_clusters(S_R_5)

        risk_score_D = Indicators.risk_web(S_R_D, CURRENT_ASK)
        risk_score_M15 = Indicators.risk_web(S_R_15, CURRENT_ASK)
        risk_score_M5 = Indicators.risk_web(S_R_5, CURRENT_ASK)

        range_score_M5 = Indicators.get_range(S_R_5)
        range_score_M15 = Indicators.get_range(S_R_15)
        range_score_D = Indicators.get_range(S_R_D)

        #support and resistance ranges when its doing well, trend indicator when its doing well, risk score when its doing well
        breakout_bool = False
        # Find minimum and maximum prices in each half
        min_top_half_M15 = min(top_half_M15)
        max_bottom_half_M15 = max(bottom_half_M15)
        min_top_half_M5 = min(top_half_M5)
        max_bottom_half_M5 = max(bottom_half_M5)
        min_top_half_D = min(top_half_D)
        max_bottom_half_D = max(bottom_half_D)

        #print(RSI_14_D)
        # Determine whether to buy, sell, or hold

        if CURRENT_ASK < max_bottom_half_M5 and risk_score_D < 0.7:
            if CURRENT_ASK < max_bottom_half_M15 and risk_score_M5 > 0.5:
                if CURRENT_ASK < EMA_50 and CURRENT_ASK > EMA_9:
                    sd.SELL = True
                    sd.BUY = False
        if CURRENT_ASK > min_top_half_M5 and risk_score_D < 0.7:
            if CURRENT_ASK > min_top_half_M15 and risk_score_M5 > 0.5:
                if CURRENT_ASK > EMA_50 and CURRENT_ASK < EMA_9:
                    sd.BUY = True
                    sd.SELL = False
                    
        if not self.simulation:
            self.snapshot = {
                'risk_score_D': risk_score_D,
                'risk_score_M15': risk_score_M15,
                'risk_score_M5': risk_score_M5,
                'range_score_M5': range_score_M5,
                'range_score_M15': range_score_M15,
                'range_score_D': range_score_D,
                'RSI_14_D': RSI_14_D,
                'time' : self.current_timestamp,
                'sd.BUY' : sd.BUY,
                'sd.SELL' : sd.SELL,
                'breakout' : breakout_bool
            }

            #self.algorithm_tools.log_to_main(f'>> snapshot @ : {self.snapshot}')
            
        if TRADE_COUNT == 0:

            if not breakout_bool:
                if self.SELL:
                    if RSI_14_D > 70:
                        return
                    
                if self.BUY:
                    if RSI_14_D < 30:
                        return
                sd.add_price = 0
                sd.prompted = False
                sd.original = 0
                sd.phone = False
                sd.add_price_increment = 0.0025
                position_size = size_tools.saw_with_news_impact(sim, EQUITY, sd.original, current_time=self.current_timestamp, amplitude=1, offset=5)
                
                
                if self.BUY:
                    self.algorithm_tools.log_to_main(f'>> running time @ : {self.current_timestamp}')
                    self.algorithm_tools.log_to_main(f'>> HIGH RISK LONG POINT OF INTEREST @ : {CURRENT_ASK}')
                    self.algorithm_tools.log_to_main(f'>> REALIZED @ : {REALIZED}')
                    self.algorithm_tools.log_to_main(f'>> DRAWDOWN @ : {DRAWDOWN}')
                    self.api.place_trade(self.symbol, position_size, 1, sim, sd.original, CURRENT_ASK, current_time)
                    sd.original += 1
                    sd.add_price = CURRENT_ASK
                if self.SELL:
                    self.algorithm_tools.log_to_main(f'>> running time @ : {self.current_timestamp}')
                    self.algorithm_tools.log_to_main(f'>> HIGH RISK SHORT POINT OF INTEREST @ : {CURRENT_ASK}')
                    self.algorithm_tools.log_to_main(f'>> REALIZED @ : {REALIZED}')
                    self.algorithm_tools.log_to_main(f'>> DRAWDOWN @ : {DRAWDOWN}')
                    self.api.place_trade(self.symbol, position_size, -1, sim, sd.original, CURRENT_ASK, current_time)
                    sd.original += 1
                    sd.add_price = CURRENT_ASK

        if TRADE_COUNT > 0:
            
            risk_model, cos_, prev_date, next_date, total_seconds = time_tools.calculate_news_impact(sim, current_time=self.current_timestamp)
            LONG = a_pos['long']
            SHORT = a_pos['short']
            units_l = int(LONG['units'])
            units_s = int(SHORT['units'])

            if sd.original == 0:
                sd.original = TRADE_COUNT
                sd.add_price = float(self.api.get_last_order_price(acc=sim))
                print(sd.add_price)
                print(sd.original)
                print(TRADE_COUNT)

                if UNREALIZED > 0:
                    self.algorithm_tools.log_to_main(f'UNREALIZED PROFIT: {UNREALIZED}')
                    self.algorithm_tools.log_to_main(f'CLOSE $: {CURRENT_ASK}')
                    self.algorithm_tools.log_to_main(f'REALIZED_PERC $: {REALIZED_PERC}')
                    self.api.close_position(self.symbol, long_units=units_l, acc=sim, unrealized=UNREALIZED, ask=CURRENT_ASK, current_time=current_time)
            
            if MARGIN_CLOSEOUT_PERCENT > 0.10:
                print(MARGIN_CLOSEOUT_PERCENT)
                message=f"Margin closeout percent is {MARGIN_CLOSEOUT_PERCENT}%, do you want to continue trading? (y/n): "
                self.algorithm_tools.phone_alert(msg=message)
                if not sd.prompted:
                    answer = input(f"Margin closeout percent is {MARGIN_CLOSEOUT_PERCENT}%, do you want to continue trading? (y/n): ")
                    if answer.lower() == "n":
                        raise Exception("Trading halted due to high margin closeout percent.")
                    elif answer.lower() == "y":
                        sd.prompted = True
                return
            if MARGIN_CLOSEOUT_PERCENT > 0.25:
                if UNREALIZED > round(EQUITY * -0.05, 2):
                    self.algorithm_tools.log_to_main(f'UNREALIZED PROFIT: {UNREALIZED}')
                    self.algorithm_tools.log_to_main(f'CLOSE $: {CURRENT_ASK}')
                    self.algorithm_tools.log_to_main(f'REALIZED_PERC $: {REALIZED_PERC}')
                    self.api.close_position(self.symbol, long_units=units_l, acc=sim, unrealized=UNREALIZED, ask=CURRENT_ASK, current_time=current_time)
                    return

            for i in range(len(CLOSE)):
                if UNREALIZED > CLOSE[i] and sd.original >= i:
                    self.algorithm_tools.log_to_main(f'UNREALIZED PROFIT: {UNREALIZED}')
                    self.algorithm_tools.log_to_main(f'CLOSE $: {CURRENT_ASK}')
                    self.algorithm_tools.log_to_main(f'REALIZED_PERC $: {REALIZED_PERC}')
                    self.api.close_position(self.symbol, long_units=units_l, acc=sim, unrealized=UNREALIZED, ask=CURRENT_ASK, current_time=current_time)
                    break 
                
                #add more positions logic here

    def OnEndOfAlgorithm(self):
        sim = self.simulation_data[self.symbol]
        # Create Plotly chart

        fig = make_subplots(rows=3, 
                            cols=2, 
                            subplot_titles=[
                                'losing_candles',
                                'sim_NAV',
                                'sim_realized_perc',
                                'margin_closeout_perc',
                                'max_drawdown',
                                'CURRENT_ASK',
                            ], 
                            row_heights=[500, 500, 500], 
                            vertical_spacing=0.05,
                            column_widths=[0.9, 0.9]
                            )

        color_blue = '#0c2a4d'
        color_red = '#b30000'
        color_green = '#2e8b57'
        color_orange = '#b35900'
        color_purple = '#511845'
        color_pink = '#c70039'
        color_yellow = '#ffd500'
        color_gray = '#808080'
        color_white = '#ffffff'

        fig.update_layout(
            paper_bgcolor='#101010',  # set background color of paper to black
            plot_bgcolor='#101010',  # set background color of plot to black
            font_color='#d6d6d6',  # set font color to off-white
            title_font_color='#d6d6d6',  # set title font color to off-white
            modebar_color='#d6d6d6',  # set modebar color to off-white
            template='plotly_dark',  # enable dark mode
            hoverlabel=dict(bgcolor='#333333'),  # set hover label background color
            xaxis=dict(linecolor=color_gray),  # set x-axis color to gray
            yaxis=dict(linecolor=color_gray),  # set y-axis color to gray
            legend=dict(bgcolor='#101010', bordercolor=color_white, borderwidth=1),  # set legend background color to black
        )

        
        for i, key in enumerate(self.simulation_results[self.symbol].keys()):
            if key == 'time':
                continue
            elif key == 'losing_candles':
                # Add losing_candles as a scatter plot
            
                sim_data = self.simulation_results.get(self.symbol, {})
                fig.add_trace(go.Scatter(x=sim_data[key]['time'], 
                                         y=sim_data[key]['mid_c'], 
                                         mode='lines', 
                                         name=key,
                                         line=dict(color=color_white)), 
                              row=1, col=1)
                # Add horizontal lines for each trade
                for trade_num, trade_data in sim.sim_long.items():
                    trade_price = trade_data['price']
                    trade_units = trade_data['units']
                    y_values = [trade_price] * len(sim_data['time'])
                    fig.add_trace(go.Scatter(x=sim_data['time'], y=y_values, 
                                             mode='lines', line_shape='hv', 
                                             name=f'Long Trade {trade_num} ({trade_units} units)', 
                                             line=dict(dash='dash', width=1, color=color_yellow)), 
                                  row=1, col=1)
                for trade_num, trade_data in sim.sim_short.items():
                    trade_price = trade_data['price']
                    trade_units = trade_data['units']
                    y_values = [trade_price] * len(sim_data['time'])
                    fig.add_trace(go.Scatter(x=sim_data['time'], y=y_values, 
                                             mode='lines', line_shape='hv', 
                                             name=f'Short Trade {trade_num} ({trade_units} units)', 
                                             line=dict(dash='dash', width=1, color=color_pink)), 
                                  row=1, col=1)
            elif key == 'CURRENT_ASK':
                # Add losing_candles as a scatter plot
                sim_data = self.simulation_results.get(self.symbol, {})
                fig.add_trace(go.Scatter(x=sim_data['time'], 
                                        y=sim_data[key], 
                                        mode='lines', 
                                        name=key,
                                        line=dict(color=color_white)), 
                            row=3, col=2)

                # Add scatter dots for each long trade
                for trade_num, trade_pos in enumerate(sim.sim_long_trades):
                    for trade_id, trade_data in trade_pos.items():
                        trade_price = trade_data['price']
                        trade_units = trade_data['units']
                        trade_time = trade_data['time']
                        trade_close = trade_data['close_time']
                        trade_close_price = trade_data['close_price']
                        fig.add_trace(go.Scatter(x=[trade_time, trade_close], y=[trade_price, trade_close_price], 
                                                mode='markers+lines', line_shape='hv', 
                                                name=f'Long Trade {trade_num} ({trade_units} units)', 
                                                line=dict(width=1, color=color_yellow),
                                                showlegend=False), row=3, col=2)

                # Add scatter dots for each short trade
                for trade_num, trade_pos in enumerate(sim.sim_short_trades):
                    for trade_id, trade_data in trade_pos.items():
                        trade_price = trade_data['price']
                        trade_units = trade_data['units']
                        trade_time = trade_data['time']
                        trade_close = trade_data['close_time']
                        trade_close_price = trade_data['close_price']
                        fig.add_trace(go.Scatter(x=[trade_time, trade_close], y=[trade_price, trade_close_price], 
                                                mode='markers+lines', line_shape='hv', 
                                                name=f'Short Trade {trade_num} ({trade_units} units)', 
                                                line=dict(width=1, color=color_pink),
                                                showlegend=False), row=3, col=2)

            elif key == 'sim_NAV':
                sim_data = self.simulation_results.get(self.symbol, {})
                fig.add_trace(go.Scatter(x=sim_data['time'], y=sim_data[key], name=key, line=dict(color=color_white)), row=1, col=2)
            elif key == 'sim_realized_perc':
                sim_data = self.simulation_results.get(self.symbol, {})
                fig.add_trace(go.Scatter(x=sim_data['time'], y=sim_data[key], name=key, line=dict(color=color_white)), row=2, col=1)
            elif key == 'margin_closeout_perc':
                sim_data = self.simulation_results.get(self.symbol, {})
                fig.add_trace(go.Scatter(x=sim_data['time'], y=sim_data[key], name=key, line=dict(color=color_white)), row=2, col=2)
            elif key == 'max_drawdown':
                sim_data = self.simulation_results.get(self.symbol, {})
                fig.add_trace(go.Scatter(x=sim_data['time'], y=sim_data[key], name=key, line=dict(color=color_white)), row=3, col=1)

            
        fig.update_layout(title=f"Simulation Results for {self.symbol}")
        fig.update_yaxes(title_text="Values", tickprefix="$", row=1, col=1)
        fig.write_html(f'simulation_results_{time.time()}.html', auto_open=True)
        print('Plot saved to file: simulation_results.html')

    def run(self, restart=True):

        sd = self.symbolData[self.symbol]
        live = False

        time_frames = ['M5', 'M15', 'D']

        # Log current buy/sell decision
        self.algorithm_tools.log_to_main(f"Buy: {self.BUY}, Sell: {self.SELL}")

        if not self.simulation:
            self.current_timestamp = datetime.datetime.now()
            scheduler1 = schedule.Scheduler()
            scheduler1.every(15).minutes.do(lambda: self.collect_candles(granularities=time_frames))
            

            while True:
                live = time_tools.check_weekday()
                time.sleep(handfruit_hedge.SLEEP)
                if live:
                    scheduler1.run_pending()
                    try:
                        self.OnData()
                    except Exception as error:
                        self.algorithm_tools.log_to_error(f"CRASH: {error}")
        else:
            sim = self.simulation_data[self.symbol]
            closeout_terminate = sim.margin_closeout_perc
            simulation_starting_time_str = sim.start.strftime("%Y-%m-%dT%H:%M:%SZ")
            simulation_starting_time = datetime.datetime.strptime(simulation_starting_time_str, "%Y-%m-%dT%H:%M:%SZ")
            warm_up_period_days = 20
            warm_up_period_end = simulation_starting_time + datetime.timedelta(days=warm_up_period_days)
            current_time = simulation_starting_time


            for i in range(len(sd.candles['M5'])):
                self.current_timestamp = sd.candles['M5'].iloc[i]['time']
                current_str = self.current_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
                current_time = datetime.datetime.strptime(current_str, "%Y-%m-%dT%H:%M:%SZ")

                # Check margin closeout percentage
                sim = self.simulation_data[self.symbol]
                closeout_terminate = sim.margin_closeout_perc
                """
                if closeout_terminate > 0.45:
                    self.algorithm_tools.log_to_main(f"Margin closeout percentage exceeded 35%: {closeout_terminate}")
                    break
                """
                if current_time >= warm_up_period_end:
                    try:
                        self.OnData()
                    except Exception as error:
                        self.algorithm_tools.log_to_error(f"CRASH: {error}")
            self.OnEndOfAlgorithm()
            log_message = f"Simulation ended at {current_time} : snapshot tools: {self.top_five}"
            self.algorithm_tools.log_to_main(log_message)
            self.algorithm_tools.log_to_main(self.snapshot)
            # Create a new file named "log.txt" and write the log message to it
            with open("favorable.txt", "w") as f:
                f.write(log_message)
            # End of simulation
            




    