import requests
import pandas as pd
import json
import constants.defs as defs
import time

from dateutil import parser
from datetime import datetime as dt
#from infrastructure.instrument_collection import instrumentCollection as ic
from models.api_price import ApiPrice
#from models.open_trade import OpenTrade

class OandaApiSim:

    def __init__(self, api_key=None, oanda_url='https://api-fxtrade.oanda.com/v3', account_id=None):
        self.api_key = api_key
        self.oanda_url = oanda_url
        self.account_id = account_id
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def make_request(self, url, verb='get', code=200, params=None, data=None, headers=None):
        full_url = f"{self.oanda_url}/{url}"

        if data is not None:
            data = json.dumps(data)

        for i in range(3): # try 3 times
            try:
                response = None
                if verb == "get":
                    response = self.session.get(full_url, params=params, data=data, headers=headers)
                elif verb == "post":
                    response = self.session.post(full_url, params=params, data=data, headers=headers)
                elif verb == "put":
                    response = self.session.put(full_url, params=params, data=data, headers=headers)

                if response == None:
                    return False, {'error': 'verb not found'}

                if response.status_code == code:
                    return True, response.json()
                else:
                    return False, response.json()
            except Exception as error:
                if "openPositions" in url:
                    return False, {'error': 'unexpected error'}
                if i < 2: # if it's not the last try
                    print(f"Request failed, retrying... (attempt {i+1})")
                    continue
                else:
                    return False, {'error': error}



    def get_accounts(self, data_key):
        url = f"accounts"
        ok, data = self.make_request(url)
        if ok == True and data_key in data:
            return data[data_key]
        else:
            print("ERROR get_account_ep()", data)
            return None

    def get_account_ep(self, ep, data_key, acc_id):
        url = f"accounts/{acc_id}/{ep}"
        ok, data = self.make_request(url)

        if ok == True and data_key in data:
            return data[data_key]
        else:
            print("ERROR get_account_ep()", data)
            return None

    def get_account_ids(self):
        acc_ids = []
        accs = self.get_accounts("accounts")
        accs_len = len(accs)
        for i in range(accs_len):
            id = accs[i]['id']
            acc_ids.append(id)
        return acc_ids

    def get_account_summary(self, account_id, acc=None):
        if acc is None:
            return self.get_account_ep("summary", "account", account_id)
        else:
            return {'unrealizedPL': acc.sim_unrealized, 'NAV': acc.sim_NAV, 'openTradeCount': acc.sim_trade_count}

    
    def get_position_summary(self, account_id, acc=None):
        if acc is None:
            data = self.get_account_ep("openPositions", "positions", account_id)
            if isinstance(data, list) and data:
                try:
                    data_dic = data[0]
                except IndexError:
                    return None
                return data_dic
            else:
                return None
        else:
            long_units = sum([details['units'] for count, details in acc.sim_long.items()])
            short_units = sum([details['units'] for count, details in acc.sim_short.items()])

            if long_units == 0 and short_units == 0:
                return None
            else:
                long_pos = {'units': long_units}
                short_pos = {'units': short_units}
                return {'long': long_pos, 'short': short_pos}

    def get_account_instruments(self, account_id):
        return self.get_account_ep("instruments", "instruments", account_id)
        
    def fetch_candles(self, pair_name, count=10, granularity="H1",
                            price="MBA", date_f=None, date_t=None):
        url = f"instruments/{pair_name}/candles"
        params = dict(
            granularity = granularity,
            price = price
        )

        if date_f is not None and date_t is not None:
            date_format = "%Y-%m-%dT%H:%M:%SZ"
            params["from"] = dt.strftime(date_f, date_format)
            params["to"] = dt.strftime(date_t, date_format)
        else:
            params["count"] = count

        ok, data = self.make_request(url, params=params)

        if ok == True and 'candles' in data:
            return data['candles']
        else:
            print("ERROR fetch_candles()", params, data)
            return None

    def get_candles_df(self, pair_name, **kwargs):

        data = self.fetch_candles(pair_name, **kwargs)

        if data is None:
            return None
        if len(data) == 0:
            return pd.DataFrame()
        
        prices = ['mid', 'bid', 'ask']
        ohlc = ['o', 'h', 'l', 'c']
        
        final_data = []
        for candle in data:
            if candle['complete'] == False:
                continue
            new_dict = {}
            new_dict['time'] = parser.parse(candle['time'])
            new_dict['volume'] = candle['volume']
            for p in prices:
                if p in candle:
                    for o in ohlc:
                        new_dict[f"{p}_{o}"] = float(candle[p][o])
            final_data.append(new_dict)
        df = pd.DataFrame.from_dict(final_data)
        return df

    def last_complete_candle(self, pair_name, granularity):
        df = self.get_candles_df(pair_name, granularity=granularity, count=10)
        if df.shape[0] == 0:
            return None
        return df.iloc[-1].time

    def get_ask_bid(self, instruments_list, t=None, candles_df=None, sim_bool=False):
        if sim_bool:
            timestamp = t
            try:
                row = candles_df.loc[candles_df['time'] == timestamp]
                ask = row.iat[0, row.columns.get_loc('mid_c')]
                return ask
            except:
                print("ERROR get_ask_bid()")
                return None

        else:
            url = f"accounts/{self.account_id}/pricing"

            params = dict(
                instruments=','.join(instruments_list),
                includeHomeConversions=True
            )

            ok, response = self.make_request(url, params=params)

            if ok == True and 'prices' in response and 'homeConversions' in response:
                pri = [ApiPrice(x, response['homeConversions']) for x in response['prices']]
                p = pri[0]
                ask = p.ask
                bid = p.bid
                return ask, bid

            return None


    def get_close(self):
        url = f"accounts/{self.account_id}/candles/latest"

        params = dict(
            candleSpecifications='EUR_USD:S5:BM'
        )
        ok, response = self.make_request(url, params=params)
        
        if ok == True and 'latestCandles' in response:
            lc = response['latestCandles']
            prices = lc[0]
            candles = prices['candles']
            c = candles[0]
            return c['bid']
        return None

    def get_filled_orders(self, account_id):
        order_info = {}
        url = f"accounts/{account_id}/orders"

        params = dict(
            state='ALL',
            count=1
        )

        ok, response = self.make_request(url, verb="get", code=200, params=params)

        if 'orders' in response:
            order_response = response['orders']
            o = order_response[0]
            state = o['state']

            order_info['units'] = o['units']
            order_info['id'] = o['id']
            order_info['state'] = state


            if ok == True and state == 'FILLED':
                return order_info
            else:
                return None
    
    def get_last_order_price(self, acc):
        if acc is None:
            url = f"accounts/{self.account_id}/openTrades"

            ok, response = self.make_request(url, verb="get", code=200)
            if 'trades' in response:
                trade_response = response['trades']
                if trade_response:
                    o = trade_response[0]
                    price = o['price']
                    return price
            return None
        else:
            if acc.sim_trade_count > 0:
                last_trade_key = str(acc.sim_trade_count)
                if last_trade_key in acc.sim_long:
                    trade_info = acc.sim_long[last_trade_key]
                    price = trade_info['price']
                    return price
                elif last_trade_key in acc.sim_short:
                    trade_info = acc.sim_short[last_trade_key]
                    price = trade_info['price']
                    return price
            return None
    

    
    def calculate_weighted_average_price(self, acc):
        long_units = sum([details['units'] for count, details in acc.sim_long.items()])
        short_units = sum([details['units'] for count, details in acc.sim_short.items()])

        if long_units + short_units == 0:
            return None
        elif long_units > 0:
            total_units = long_units
            weighted_price = sum([details['units'] * details['price'] for count, details in acc.sim_long.items()]) / total_units
        elif short_units < 0:
            total_units = abs(short_units)
            weighted_price = -sum([details['units'] * details['price'] for count, details in acc.sim_short.items()]) / total_units
        else:
            return None

        return weighted_price

    def calculate_unrealized_profit_loss(self, acc, wp, ask):
        if wp is None:
            return 0

        long_units = sum([details['units'] for count, details in acc.sim_long.items()])
        short_units = sum([details['units'] for count, details in acc.sim_short.items()])
        spread = 0.00014
        margin_used = 0

        if long_units > 0:
            acc.sim_unrealized = long_units * (ask - wp - spread)
            margin_used = abs(long_units) * wp / acc.leverage
        elif short_units < 0:
            acc.sim_unrealized = short_units * (ask - wp - spread)
            margin_used = abs(short_units) * wp / acc.leverage
        else:
            acc.sim_unrealized = 0
            
        acc.sim_NAV = acc.sim_balance + acc.sim_unrealized
        acc.sim_margin_used = margin_used

        return round(acc.sim_unrealized, 2)




    def place_trade(self, pair_name: str, units: float, direction: int, acc=None, count=None, ask=None, current_time=None):
        url = f"accounts/{self.account_id}/orders"

        units = round(units, 0)

        if direction == defs.SELL:
            units = units * -1

        if acc is None:
            data = dict(
                order=dict(
                    units=str(units),
                    instrument=pair_name,
                    type="MARKET"
                )
            )
            ok, response = self.make_request(url, verb="post", data=data, code=201)

            if ok == True and 'orderFillTransaction' in response:
                return response['orderFillTransaction']['id']
            else:
                return None
        else:
            spread = 0.00014
            if direction == defs.SELL:
                ask -= spread
            else:
                ask += spread
            # Place trade on simulation account
            acc.sim_trade_count = count+1
            count = str(count)
            if direction == defs.BUY:
                acc.sim_long[count] = {'units': units, 'price': ask, 'time': current_time, 'close_time': None, 'close_price': None}
            elif direction == defs.SELL:
                acc.sim_short[count] = {'units': units, 'price': ask, 'time': current_time, 'close_time': None, 'close_price': None}



    def close_position(self, pair_name: str, long_units: int, acc=None, unrealized=None, ask=None, current_time=None):
        if acc is None:
            url = f"accounts/{self.account_id}/positions/{pair_name}/close"

            if long_units > 0:
                data = dict(
                    longUnits='ALL'
                )
            else:
                data = dict(
                    shortUnits='ALL'
                )
            ok, response = self.make_request(url, verb="put", data=data, code=200)

            if ok == True and 'relatedTransactionIDs' in response:
                return response['relatedTransactionIDs']
            else:
                return None
        else:
            # Close trade on simulation account
            

            
            if long_units > 0:
                for count in acc.sim_long.keys():
                    acc.sim_long[count]['close_time'] = current_time
                    acc.sim_long[count]['close_price'] = ask
                acc.sim_long_trades.append(acc.sim_long.copy())
                acc.sim_long.clear()
            else:
                for count in acc.sim_short.keys():
                    acc.sim_short[count]['close_time'] = current_time
                    acc.sim_short[count]['close_price'] = ask
                acc.sim_short_trades.append(acc.sim_short.copy())
                acc.sim_short.clear()


            acc.sim_trade_count = 0
            acc.sim_realized += unrealized
            acc.sim_balance += unrealized
            acc.sim_unrealized = 0
            acc.sim_margin_used = 0

            # Calculate the realized percentage gain/loss
            realized_perc = (acc.sim_balance - acc.starting_balance) / acc.starting_balance * 100
            acc.sim_realized_perc = realized_perc
