import pandas as pd
import datetime as dt
from dateutil import parser

from infrastructure.instrument_collection import InstrumentCollection
from api.oanda_api_sim import OandaApiSim

CANDLE_COUNT = 1000

INCREMENTS = {
    'S5' : 0.05 * CANDLE_COUNT,
    'M1' : 1 * CANDLE_COUNT,
    'M5' : 5 * CANDLE_COUNT,
    'M15' : 15 * CANDLE_COUNT,
    'H1' : 60 * CANDLE_COUNT,
    'H4' : 240 * CANDLE_COUNT,
    'D' : 1440 * CANDLE_COUNT,
}


def save_file(final_df: pd.DataFrame, file_prefix, granularity, pair):
    filename = f"{file_prefix}{pair}_{granularity}.pkl"

    final_df.drop_duplicates(subset=['time'], inplace=True)
    final_df.sort_values(by='time', inplace=True)
    final_df.reset_index(drop=True, inplace=True)
    final_df.to_pickle(filename);

    s1 = f"*** {pair} {granularity} {final_df.time.min()} {final_df.time.max()}"
    print(f"*** {s1} --> {final_df.shape[0]} candles ***")


def fetch_candles(pair, granularity, date_f: dt.datetime, 
                    date_t: dt.datetime, api: OandaApiSim ):

    attempts = 0

    while attempts < 3:

        candles_df = api.get_candles_df(
            pair,
            granularity=granularity,
            date_f=date_f,
            date_t=date_t
        )

        if candles_df is not None:
            break

        attempts += 1

    if candles_df is not None and candles_df.empty == False:
        return candles_df
    else:
        return None


def collect_data(pair, granularity, date_f, date_t, file_prefix, api: OandaApiSim ):
    
    time_step = INCREMENTS[granularity]

    end_date = parser.parse(date_t)
    from_date = parser.parse(date_f)

    candle_dfs = []

    to_date = from_date

    while to_date < end_date:
        to_date = from_date + dt.timedelta(minutes=time_step)
        if to_date > end_date:
            to_date = end_date

        candles = fetch_candles(
            pair,
            granularity, 
            from_date,
            to_date,
            api
        )

        if candles is not None:
            candle_dfs.append(candles)
            print(f"{pair} {granularity} {from_date} {to_date} --> {candles.shape[0]} candles loaded")
        else:
            print(f"{pair} {granularity} {from_date} {to_date} --> NO CANDLES")
        
        from_date = to_date
    
    if len(candle_dfs) > 0:
        final_df = pd.concat(candle_dfs)
        save_file(final_df, file_prefix, granularity, pair)
    else:
        print(f"{pair} {granularity} --> NO DATA SAVED!")



def run_collection(ic: InstrumentCollection, api: OandaApiSim):
    our_curr = ["USD", "EUR"]
    for p1 in our_curr:
        for p2 in our_curr:
            pair = f"{p1}_{p2}"
            if pair in ic.instruments_dict.keys():
                for granularity in ['M1']:
                    print(pair, granularity)
                    collect_data(
                        pair,
                        granularity,
                        "2021-01-01T00:00:00Z",
                        "2023-01-01T00:00:00Z",
                        "./data/",
                        api
                    )

























