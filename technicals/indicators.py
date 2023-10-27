import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from numpy import arange
from scipy import stats
class Indicators:


    def __init__(self, symbol):
        self.symbol = symbol
        self.ind = {}

    def smooth(arr, window, order):
        win = np.ones(int(window)) / float(window)
        smoothed = np.convolve(arr, win, mode='same')
        for i in range(order - 1):
            smoothed = np.convolve(smoothed, win, mode='same')
            return smoothed
        
    def cluster(data, maxgap):
        if len(data) == 0:
            return []
        else:
            data.sort()
            groups = [[data[0]]]
            for x in data[1:]:
                if abs(x - groups[-1][-1]) <= maxgap:
                    groups[-1].append(x)
                else:
                    groups.append([x])
            return groups
        
    def array_cluster(data, maxgap):
        if data.size == 0:
            return np.array([])
        else:
            data.sort()
            groups = np.array([[data[0]]])
            for x in data[1:]:
                if abs(x - groups[-1][-1]) <= maxgap:
                    groups[-1] = np.append(groups[-1], x)
                else:
                    groups = np.append(groups, np.array([[x]]))
            return groups

    def RSI(df: pd.DataFrame, n=14):
        alpha = 1.0 / n
        gains = df.mid_c.diff()

        wins = pd.Series([ x if x >= 0 else 0.0 for x in gains ], name="wins")
        losses = pd.Series([ x * -1 if x < 0 else 0.0 for x in gains ], name="losses")

        wins_rma = wins.ewm(min_periods=n, alpha=alpha).mean()
        losses_rma = losses.ewm(min_periods=n, alpha=alpha).mean()

        rs = wins_rma / losses_rma

        df[f"RSI_{n}"] = 100.0 - (100.0 / (1.0 + rs))
        return df

    def EMA(df: pd.DataFrame, n=50):
        df[f'EMA_{n}'] = df.mid_c.ewm(span=n, min_periods=n).mean()
        return df

    def ATR(df: pd.DataFrame, n=14):
        df['H-L'] = abs(df['High'] - df['Low'])
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
        df['ATR'] = df['TR'].rolling(n).mean()
        df.drop(['H-L', 'H-PC', 'L-PC', 'TR'], inplace=True, axis=1)
        return df


    def mm(df: pd.DataFrame, n=int, pip_range=float):
        """ input: df=prices df, n=lookback, pip_range= s and r zone width
            return: support and resistance zones as a list
        """
        mm_list = df['mid_c'].values.tolist()
        np_mm = np.array(mm_list)
        shape = np_mm.shape[0]
        smooth_ = savgol_filter(np_mm, (n + 1), 3)
        dp = np.zeros(shape)
        dp[1:] = np.subtract(smooth_[1:], smooth_[:-1])
        minima = []
        maxima = []
        for i in range(shape - n):
            s = dp[i:(i+n)]
            first = s[:int((n / 2))]
            last = s[int((n  / 2)):]
            r_1 = np.sum(first > 0)
            r_2 = np.sum(last < 0)
            s_1 = np.sum(first < 0)
            s_2 = np.sum(last > 0)
            if (r_1 == int((n / 2))) and (r_2 == int((n / 2))):
                maxima.append(np_mm[i + (int((n / 2)) - 1)])
            if (s_1 == (n / 2)) and (s_2 == (n / 2)):
                minima.append(np_mm[i + (int((n / 2)) - 1)])

        s_r = minima+maxima
        g = Indicators.cluster(s_r, pip_range)
        return g
    def mm_collect(df: pd.DataFrame, n=int, pip_range=float):
        """ input: df=prices df, n=lookback, pip_range= s and r zone width
            return: support and resistance zones as a list
        """
        mm_list = df['mid_c'].values.tolist()
        np_mm = np.array(mm_list)
        shape = np_mm.shape[0]
        smooth_ = savgol_filter(np_mm, (n + 1), 3)
        dp = np.zeros(shape)
        dp[1:] = np.subtract(smooth_[1:], smooth_[:-1])
        minima = []
        maxima = []
        for i in range(shape - n):
            s = dp[i:(i+n)]
            first = s[:int((n / 2))]
            last = s[int((n  / 2)):]
            r_1 = np.sum(first > 0)
            r_2 = np.sum(last < 0)
            s_1 = np.sum(first < 0)
            s_2 = np.sum(last > 0)
            if (r_1 == int((n / 2))) and (r_2 == int((n / 2))):
                maxima.append(np_mm[i + (int((n / 2)) - 1)])
            if (s_1 == (n / 2)) and (s_2 == (n / 2)):
                minima.append(np_mm[i + (int((n / 2)) - 1)])

        s_r = minima+maxima
        g = Indicators.cluster(s_r, pip_range)
        print(df['time'].iloc[-1])
        print(g)
        return g
    
    def sup_res(df: pd.DataFrame, n=int, pip_range=float, periods=int, granularity=str ):
        """ input: df=prices df, n=lookback, pip_range= s and r zone width
            return: support and resistance zones as a list
        """
        df[f'support_resistance_{granularity}'] = np.nan
        for i, row in df.iterrows():
            if i > periods:
                rows = df.copy().reset_index(drop=True).iloc[i-periods:i]
                mm_list = rows['mid_c'].values.tolist()
                np_mm = np.array(mm_list, dtype=object)
                shape = np_mm.shape[0]
                if n > shape:
                    raise ValueError("The value of n cannot be greater than the length of np_mm")
                smooth_ = savgol_filter(np_mm, (n + 1), 3)
                dp = np.zeros(shape)
                dp[1:] = np.subtract(smooth_[1:], smooth_[:-1])
                minima = []
                maxima = []
                for j in range(shape - n):
                    s = dp[j:(j+n)]
                    first = s[:int((n / 2))]
                    last = s[int((n  / 2)):]
                    r_1 = np.sum(first > 0)
                    r_2 = np.sum(last < 0)
                    s_1 = np.sum(first < 0)
                    s_2 = np.sum(last > 0)
                    if (r_1 == int((n / 2))) and (r_2 == int((n / 2))):
                        maxima.append(np_mm[j + (int((n / 2)) - 1)])
                    if (s_1 == (n / 2)) and (s_2 == (n / 2)):
                        minima.append(np_mm[j + (int((n / 2)) - 1)])
                s_r = minima + maxima
                g = Indicators.cluster(s_r, pip_range)
                df.at[i, f'support_resistance_{granularity}'] = g
        return df






    #validate with 3rd point.
    #add parallel channel marker by comparing min and max

    def get_trendline_point(df: pd.DataFrame, n=int):
        """ input: df=prices df, n=lookback, 
            return: trendline tuple with min and max trenline prices
        """
        max_idx = argrelextrema(df["mid_h"].values, np.greater, order=n)[0]
        min_idx = argrelextrema(df["mid_l"].values, np.less, order=n)[0]
        
        cc_max = df.iloc[max_idx][['mid_o', 'mid_c', 'mid_h', 'mid_l']]
        cc_min = df.iloc[min_idx][['mid_o', 'mid_c', 'mid_h', 'mid_l']]

        max_index = cc_max.index.tolist()
        min_index = cc_min.index.tolist()

        max_arrange = arange(0,len(max_index))
        min_arrange = arange(0,len(min_index))

        trendlines_max = []
        trendlines_min = []

        for i in max_arrange:
            if i < len(max_index)-1:
                j = i+1
                k_1 =  [cc_max.index[i],cc_max.index[j]]
                y_1 =  [cc_max['mid_h'][k_1[0]],cc_max['mid_h'][k_1[1]]]

                slope, intercept, r_value, p_value, std_err = stats.linregress(k_1, cc_max['mid_h'][[max_index[i], max_index[j]]])

                x = arange(len(df.index),len(df.index)+1)
                y = slope*x+intercept
                
                trendlines_max.append(y[0])
        
        for i in min_arrange:
            if i < len(min_index)-1:
                j = i+1
                k_1 =  [cc_min.index[i],cc_min.index[j]]
                y_1 =  [cc_min['mid_l'][k_1[0]],cc_min['mid_l'][k_1[1]]]

                slope, intercept, r_value, p_value, std_err = stats.linregress(k_1, cc_min['mid_l'][[min_index[i], min_index[j]]])

                x = arange(len(df.index),len(df.index)+1)

                y = slope*x+intercept

                trendlines_min.append(y[0])


        return trendlines_max, trendlines_min


    def higher_high(df):
        df["HH"] = df["mid_c"].rolling(2).max()
        df["HL"] = ((df["mid_c"] > df["mid_c"].shift(1)) & (df["mid_c"].shift(1) < df["mid_c"].shift(2)))
        return df[df["HL"] | df["LH"]]
    
    def lower_low(df):
        df["LL"] = df["mid_c"].rolling(2).min()
        df["LH"] = ((df["mid_c"] < df["mid_c"].shift(1)) & (df["mid_c"].shift(1) > df["mid_c"].shift(2)))
        return df[df["HL"] | df["LH"]]
    
    def trend_indicator(df):
        # Call higher_high and lower_low on the data frame
        hh_df = Indicators.higher_high(df)
        ll_df = Indicators.lower_low(df)
        
        # Check if there are any trends present
        if len(hh_df) > 0:
            return "Upward trend"
        elif len(ll_df) > 0:
            return "Downward trend"
        else:
            return "No trend"
    
    def split_clusters(S_R):
        """
        Splits a list of S/R clusters into two lists containing the top and bottom half of the clusters.

        Parameters:
            S_R (list): A list of S/R clusters, where each cluster is represented as a list of numbers.
            current (float): The current asking price.

        Returns:
            A tuple containing the top half and bottom half of the clusters, respectively, as two lists of floats.
        """
        S_R_flat = [val for sublist in S_R for val in sublist]
        S_R_flat_sorted = sorted(S_R_flat, reverse=True)

        mid = len(S_R_flat_sorted) // 2
        top_half = S_R_flat_sorted[:mid]
        bottom_half = S_R_flat_sorted[mid:]

        return top_half, bottom_half

    def detect_breakout(S_R, current_price, lookback_period, breakout_threshold):
        """
        Determines if the current price has broken out of a certain range based on S/R levels.

        Parameters:
            S_R (list): A list of S/R levels, where each level is represented as a float or a list of floats.
            current_price (float): The current price of the asset.
            lookback_period (int): The number of previous bars to consider when looking for a range.
            breakout_threshold (float): The percentage by which the price must break out of the range to be considered a breakout.

        Returns:
            A boolean value indicating whether a breakout has occurred.
        """
        # Flatten S/R levels list
        S_R_values = [val if isinstance(val, float) else val[0] for val in S_R]

        # Calculate range based on lookback period
        range_high = max(S_R_values[-lookback_period:])
        range_low = min(S_R_values[-lookback_period:])

        # Calculate breakout threshold
        breakout_distance = (range_high - range_low) * (breakout_threshold / 100)

        # Check if current price is outside range
        if current_price > range_high + breakout_distance or current_price < range_low - breakout_distance:
            return True
        else:
            return False



    def calculate_volatility(gains, rolling_period):
        if len(gains) < rolling_period:
            return None
        
        recent_gains = gains[-rolling_period:]
        gains_arr = np.array(recent_gains)
        std_dev = np.std(gains_arr)
        volatility_score = std_dev / np.mean(gains_arr)

        if volatility_score > 1:
            return None

        return volatility_score

    def risk_web(S_R, ask):
        """
        Joins all the S_R list values into one list with just values and no lists, and assigns a "risk score"
        based on how close the current asking price is to the middle of the range.

        Parameters:
            S_R (list): A list of S/R clusters, where each cluster is represented as a list of numbers.
            ask (float): The current asking price.

        Returns:
            A float representing the "risk score" of the current asking price relative to the range of S/R clusters.
        """
        # Flatten S_R into a single list of values
        values = [float(val) for sublist in S_R for val in sublist]

        # Check if the current asking price is within the range of values
        if ask < min(values) or ask > max(values):
            return 1.0

        # Sort values in ascending order
        sorted_values = sorted(values)

        # Calculate the range of the values
        range_min, range_max = min(sorted_values), max(sorted_values)
        range_size = range_max - range_min

        # Calculate the current position in the range as a percentage
        position_in_range = (ask - range_min) / range_size

        # Assign a "risk score" based on the position in the range
        risk_score = 1 - abs(position_in_range - 0.5) * 2

        return risk_score


    def get_range(S_R):
        """
        Returns the range of support and resistance levels in the S_R list.

        Parameters:
            S_R (list): A list of S/R clusters, where each cluster is represented as a list of numbers.

        Returns:
            A tuple (min_val, max_val, range_val) representing the minimum and maximum values, and the range of values in the S_R list.
        """
        # Flatten S_R into a single list of values
        values = [float(val) for sublist in S_R for val in sublist]

        # Get the minimum and maximum values in the list
        min_val, max_val = min(values), max(values)

        # Get the range of values
        range_val = max_val - min_val

        return range_val