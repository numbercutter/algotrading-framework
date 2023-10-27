import datetime
from tools import time_tools
import numpy as np

def sin_with_news_impact(acc=None, equity=float, original=int, amplitude=0.5, offset=3, current_time=None):
    """
    "Calculate the number of units to be traded based on the given equity and parameters, taking into account the proximity to an upcoming news event."
    """
    fib = [
        equity * 0.1,
        equity * 0.1,
        equity * 0.1,
        equity * 0.2,
        equity * 0.2,
        equity * 0.3,
        equity * 0.3,
        equity * 0.5,
        equity * 0.8,
        equity * 1.3,
        equity * 2.1,
        equity * 3.4,
        equity * 5.5,
        equity * 8.9,
        equity * 13.4,
    ]

    if acc is None:
        current_time = datetime.datetime.now()
    else:
        current_time = current_time.replace(tzinfo=None)

    news_dates = time_tools.news_dates
    prev_date = max([d for d in news_dates if d < current_time])
    next_date = min([d for d in news_dates if d > current_time])

    total_seconds = (next_date - prev_date).total_seconds()
    elapsed_seconds = (current_time - prev_date).total_seconds()

    # Create the curved line of values
    phase = 0  # set the phase to zero, since we want the cosine wave to start at its maximum value
    # set the frequency so that the wave completes one cycle over the total time difference
    frequency = 2 * np.pi / total_seconds

    current_value = amplitude * np.cos(frequency * elapsed_seconds + phase) + offset

    units = fib[original] * current_value
    return int(units)

def saw_with_news_impact(acc=None, equity=float, original=int, amplitude=1, offset=4, current_time=None):
    """
    "Calculate the number of units to be traded based on the given equity and parameters, taking into account the proximity to an upcoming news event."
    """
    fib = [
        equity * 0.1,
        equity * 0.1,
        equity * 0.1,
        equity * 0.2,
        equity * 0.2,
        equity * 0.3,
        equity * 0.3,
        equity * 0.5,
        equity * 0.8,
        equity * 1.3,
        equity * 2.1,
        equity * 3.4,
        equity * 5.5,
        equity * 8.9,
        equity * 13.4,
    ]

    if acc is None:
        current_time = datetime.datetime.now()
    else:
        current_time = current_time.replace(tzinfo=None)

    news_dates = time_tools.news_dates
    prev_date = max([d for d in news_dates if d < current_time])
    next_date = min([d for d in news_dates if d > current_time])

    total_seconds = (next_date - prev_date).total_seconds()
    elapsed_seconds = (current_time - prev_date).total_seconds()

    # Create the curved line of values
    phase = 0  # set the phase to zero, since we want the cosine wave to start at its maximum value
    # set the frequency so that the wave completes one cycle over the total time difference
    frequency = 2 * np.pi / total_seconds

     # Create the sawtooth wave of values
    current_value = (amplitude * 2 / total_seconds) * (elapsed_seconds % (total_seconds / 2)) + offset

    units = fib[original] * current_value
    return int(units)



