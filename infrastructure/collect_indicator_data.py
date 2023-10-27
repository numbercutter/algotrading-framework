from technicals.indicators import Indicators
import numpy as np

def collect_support_resistance_data(candles_dict: dict):

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

            if granularity in ['M15', 'H4', 'D']:
                num_candles = sup_res_values.get(granularity, (0, 0))[2] if len(sup_res_values.get(granularity, (0, 0))) == 3 else 50

                candles['support_resistance'] = candles.apply(
                    lambda row: Indicators.mm_collect(
                        candles.iloc[row.name-num_candles if row.name-num_candles >=0 else 0 :row.name+1], 
                        *sup_res_values.get(granularity, (0, 0))[:2]) if row.name >= num_candles-1 else np.nan,
                    axis=1)
                filename = f"{'./data/'}_{'support_resistance'}_{'EUR_USD'}_{granularity}.pkl"
                candles.to_pickle(filename)