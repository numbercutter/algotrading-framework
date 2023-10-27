
from time import localtime, strftime
from bs4 import BeautifulSoup
import requests
import datetime
from dateutil import parser
import numpy as np

cpi_pce_ppi_2015_dates = [
    datetime.datetime(2015, 1, 16),
    datetime.datetime(2015, 2, 26),
    datetime.datetime(2015, 3, 24),
    datetime.datetime(2015, 4, 17),
    datetime.datetime(2015, 5, 22),
    datetime.datetime(2015, 6, 18),
    datetime.datetime(2015, 7, 17),
    datetime.datetime(2015, 8, 19),
    datetime.datetime(2015, 9, 16),
    datetime.datetime(2015, 10, 15),
    datetime.datetime(2015, 11, 17),
    datetime.datetime(2015, 12, 15)
]

cpi_pce_ppi_2016_dates = [
    datetime.datetime(2016, 1, 20),
    datetime.datetime(2016, 2, 19),
    datetime.datetime(2016, 3, 16),
    datetime.datetime(2016, 4, 14),
    datetime.datetime(2016, 5, 17),
    datetime.datetime(2016, 6, 16),
    datetime.datetime(2016, 7, 15),
    datetime.datetime(2016, 8, 16),
    datetime.datetime(2016, 9, 16),
    datetime.datetime(2016, 10, 18),
    datetime.datetime(2016, 11, 17),
    datetime.datetime(2016, 12, 15)
]

cpi_pce_ppi_2017_dates = [
    datetime.datetime(2017, 1, 18),
    datetime.datetime(2017, 2, 15),
    datetime.datetime(2017, 3, 15),
    datetime.datetime(2017, 4, 14),
    datetime.datetime(2017, 5, 12),
    datetime.datetime(2017, 6, 14),
    datetime.datetime(2017, 7, 14),
    datetime.datetime(2017, 8, 11),
    datetime.datetime(2017, 9, 14),
    datetime.datetime(2017, 10, 13),
    datetime.datetime(2017, 11, 15),
    datetime.datetime(2017, 12, 13)
]

cpi_pce_ppi_2018_dates = [
    datetime.datetime(2018, 1, 12),
    datetime.datetime(2018, 2, 14),
    datetime.datetime(2018, 3, 13),
    datetime.datetime(2018, 4, 11),
    datetime.datetime(2018, 5, 10),
    datetime.datetime(2018, 6, 12),
    datetime.datetime(2018, 7, 12),
    datetime.datetime(2018, 8, 10),
    datetime.datetime(2018, 9, 13),
    datetime.datetime(2018, 10, 11),
    datetime.datetime(2018, 11, 14),
    datetime.datetime(2018, 12, 12)
]

cpi_pce_ppi_2019_dates = [
    datetime.datetime(2019, 1, 11),
    datetime.datetime(2019, 2, 13),
    datetime.datetime(2019, 3, 12),
    datetime.datetime(2019, 4, 10),
    datetime.datetime(2019, 5, 9),
    datetime.datetime(2019, 6, 12),
    datetime.datetime(2019, 7, 11),
    datetime.datetime(2019, 8, 13),
    datetime.datetime(2019, 9, 12),
    datetime.datetime(2019, 10, 10),
    datetime.datetime(2019, 11, 13),
    datetime.datetime(2019, 12, 11)
]
nonfarm_payrolls_2015_dates = [
    datetime.datetime(2015, 1, 9),
    datetime.datetime(2015, 2, 6),
    datetime.datetime(2015, 3, 6),
    datetime.datetime(2015, 4, 3),
    datetime.datetime(2015, 5, 8),
    datetime.datetime(2015, 6, 5),
    datetime.datetime(2015, 7, 2),
    datetime.datetime(2015, 8, 7),
    datetime.datetime(2015, 9, 4),
    datetime.datetime(2015, 10, 2),
    datetime.datetime(2015, 11, 6),
    datetime.datetime(2015, 12, 4)
]

nonfarm_payrolls_2016_dates = [
    datetime.datetime(2016, 1, 8),
    datetime.datetime(2016, 2, 5),
    datetime.datetime(2016, 3, 4),
    datetime.datetime(2016, 4, 1),
    datetime.datetime(2016, 5, 6),
    datetime.datetime(2016, 6, 3),
    datetime.datetime(2016, 7, 8),
    datetime.datetime(2016, 8, 5),
    datetime.datetime(2016, 9, 2),
    datetime.datetime(2016, 10, 7),
    datetime.datetime(2016, 11, 4),
    datetime.datetime(2016, 12, 2)
]

nonfarm_payrolls_2017_dates = [
    datetime.datetime(2017, 1, 6),
    datetime.datetime(2017, 2, 3),
    datetime.datetime(2017, 3, 10),
    datetime.datetime(2017, 4, 7),
    datetime.datetime(2017, 5, 5),
    datetime.datetime(2017, 6, 2),
    datetime.datetime(2017, 7, 7),
    datetime.datetime(2017, 8, 4),
    datetime.datetime(2017, 9, 1),
    datetime.datetime(2017, 10, 6),
    datetime.datetime(2017, 11, 3),
    datetime.datetime(2017, 12, 8)
]

nonfarm_payrolls_2018_dates = [
    datetime.datetime(2018, 1, 5),
    datetime.datetime(2018, 2, 2),
    datetime.datetime(2018, 3, 9),
    datetime.datetime(2018, 4, 6),
    datetime.datetime(2018, 5, 4),
    datetime.datetime(2018, 6, 1),
    datetime.datetime(2018, 7, 6),
    datetime.datetime(2018, 8, 3),
    datetime.datetime(2018, 9, 7),
    datetime.datetime(2018, 10, 5),
    datetime.datetime(2018, 11, 2),
    datetime.datetime(2018, 12, 7)
]

# Federal Reserve meeting dates for 2015
fed_meetings_2015_dates = [
    datetime.datetime(2015, 1, 27),
    datetime.datetime(2015, 3, 17),
    datetime.datetime(2015, 4, 28),
    datetime.datetime(2015, 6, 16),
    datetime.datetime(2015, 7, 28),
    datetime.datetime(2015, 9, 16),
    datetime.datetime(2015, 10, 27),
    datetime.datetime(2015, 12, 15)
]

# Federal Reserve meeting dates for 2016
fed_meetings_2016_dates = [
    datetime.datetime(2016, 1, 26),
    datetime.datetime(2016, 3, 15),
    datetime.datetime(2016, 4, 26),
    datetime.datetime(2016, 6, 14),
    datetime.datetime(2016, 7, 26),
    datetime.datetime(2016, 9, 20),
    datetime.datetime(2016, 11, 1),
    datetime.datetime(2016, 12, 13)
]

# Federal Reserve meeting dates for 2017
fed_meetings_2017_dates = [
    datetime.datetime(2017, 2, 1),
    datetime.datetime(2017, 3, 15),
    datetime.datetime(2017, 5, 3),
    datetime.datetime(2017, 6, 14),
    datetime.datetime(2017, 7, 26),
    datetime.datetime(2017, 9, 20),
    datetime.datetime(2017, 11, 1),
    datetime.datetime(2017, 12, 13)
]

# Federal Reserve meeting dates for 2018
fed_meetings_2018_dates = [
    datetime.datetime(2018, 1, 31),
    datetime.datetime(2018, 3, 21),
    datetime.datetime(2018, 5, 2),
    datetime.datetime(2018, 6, 13),
    datetime.datetime(2018, 8, 1),
    datetime.datetime(2018, 9, 26),
    datetime.datetime(2018, 11, 8),
    datetime.datetime(2018, 12, 19)
]

# Federal Reserve meeting dates for 2019
fed_meetings_2019_dates = [
    datetime.datetime(2019, 1, 30),
    datetime.datetime(2019, 3, 20),
    datetime.datetime(2019, 5, 1),
    datetime.datetime(2019, 6, 19),
    datetime.datetime(2019, 7, 31),
    datetime.datetime(2019, 9, 18),
    datetime.datetime(2019, 10, 30),
    datetime.datetime(2019, 12, 11)
]
nonfarm_payrolls_2019_dates = [    datetime.datetime(2019, 1, 4),    datetime.datetime(2019, 2, 1),    datetime.datetime(2019, 3, 8),    datetime.datetime(2019, 4, 5),    datetime.datetime(2019, 5, 3),    datetime.datetime(2019, 6, 7),    datetime.datetime(2019, 7, 5),    datetime.datetime(2019, 8, 2),    datetime.datetime(2019, 9, 6),    datetime.datetime(2019, 10, 4),    datetime.datetime(2019, 11, 1),    datetime.datetime(2019, 12, 6)]

nonfarm_payrolls_2020_dates = [datetime.datetime(2020, 1, 10),    datetime.datetime(2020, 2, 7),    datetime.datetime(2020, 3, 6),    datetime.datetime(2020, 4, 3),    datetime.datetime(2020, 5, 8),    datetime.datetime(2020, 6, 5),    datetime.datetime(2020, 7, 2),    datetime.datetime(2020, 8, 7),    datetime.datetime(2020, 9, 4),    datetime.datetime(2020, 10, 2),    datetime.datetime(2020, 11, 6),    datetime.datetime(2020, 12, 4)]

cpi_pce_ppi_2020_dates = [datetime.datetime(2020, 1, 14),    datetime.datetime(2020, 2, 13),    datetime.datetime(2020, 3, 11),    datetime.datetime(2020, 4, 10),    datetime.datetime(2020, 5, 12),    datetime.datetime(2020, 6, 10),    datetime.datetime(2020, 7, 14),    datetime.datetime(2020, 8, 12),    datetime.datetime(2020, 9, 11),    datetime.datetime(2020, 10, 14),    datetime.datetime(2020, 11, 12),    datetime.datetime(2020, 12, 10)]

fed_meetings_2020_dates = [datetime.datetime(2020, 1, 28),    datetime.datetime(2020, 1, 29),    datetime.datetime(2020, 3, 3),    datetime.datetime(2020, 3, 4),    datetime.datetime(2020, 4, 28),    datetime.datetime(2020, 4, 29),    datetime.datetime(2020, 6, 9),    datetime.datetime(2020, 6, 10),    datetime.datetime(2020, 7, 28),    datetime.datetime(2020, 7, 29),    datetime.datetime(2020, 9, 15),    datetime.datetime(2020, 9, 16),    datetime.datetime(2020, 11, 4),    datetime.datetime(2020, 11, 5),    datetime.datetime(2020, 12, 15),    datetime.datetime(2020, 12, 16)]

# Nonfarm Payrolls report release dates for 2021
nonfarm_payrolls_2021_dates = [
    datetime.datetime(2021, 1, 8),
    datetime.datetime(2021, 2, 5),
    datetime.datetime(2021, 3, 5),
    datetime.datetime(2021, 4, 2),
    datetime.datetime(2021, 5, 7),
    datetime.datetime(2021, 6, 4),
    datetime.datetime(2021, 7, 2),
    datetime.datetime(2021, 8, 6),
    datetime.datetime(2021, 9, 3),
    datetime.datetime(2021, 10, 8),
    datetime.datetime(2021, 11, 5),
    datetime.datetime(2021, 12, 3)
]

# CPI/PCE/PPI report release dates for 2021
cpi_pce_ppi_2021_dates = [
    datetime.datetime(2021, 1, 13),
    datetime.datetime(2021, 2, 10),
    datetime.datetime(2021, 3, 10),
    datetime.datetime(2021, 4, 13),
    datetime.datetime(2021, 5, 12),
    datetime.datetime(2021, 6, 10),
    datetime.datetime(2021, 7, 13),
    datetime.datetime(2021, 8, 11),
    datetime.datetime(2021, 9, 14),
    datetime.datetime(2021, 10, 13),
    datetime.datetime(2021, 11, 10),
    datetime.datetime(2021, 12, 10)
]

# Federal Reserve meeting dates for 2021
fed_meetings_2021_dates = [
    datetime.datetime(2021, 1, 26),
    datetime.datetime(2021, 1, 27),
    datetime.datetime(2021, 3, 16),
    datetime.datetime(2021, 3, 17),
    datetime.datetime(2021, 4, 27),
    datetime.datetime(2021, 4, 28),
    datetime.datetime(2021, 6, 15),
    datetime.datetime(2021, 6, 16),
    datetime.datetime(2021, 7, 27),
    datetime.datetime(2021, 7, 28),
    datetime.datetime(2021, 9, 21),
    datetime.datetime(2021, 9, 22),
    datetime.datetime(2021, 11, 2),
    datetime.datetime(2021, 11, 3),
    datetime.datetime(2021, 12, 14),
    datetime.datetime(2021, 12, 15),
]

nonfarm_payrolls_2022_dates = [
    datetime.datetime(2022, 1, 7),
    datetime.datetime(2022, 2, 4),
    datetime.datetime(2022, 3, 4),
    datetime.datetime(2022, 4, 1),
    datetime.datetime(2022, 5, 6),
    datetime.datetime(2022, 6, 3),
    datetime.datetime(2022, 7, 8),
    datetime.datetime(2022, 8, 5),
    datetime.datetime(2022, 9, 2),
    datetime.datetime(2022, 10, 7),
    datetime.datetime(2022, 11, 4),
    datetime.datetime(2022, 12, 2),
    ]

cpi_pce_ppi_2022_dates = [
    datetime.datetime(2022, 1, 12), 
    datetime.datetime(2022, 2, 10), 
    datetime.datetime(2022, 3, 10),
    datetime.datetime(2022, 4, 13),
    datetime.datetime(2022, 5, 12),
    datetime.datetime(2022, 6, 10),
    datetime.datetime(2022, 7, 13),
    datetime.datetime(2022, 8, 11),
    datetime.datetime(2022, 9, 13),
    datetime.datetime(2022, 10, 13),
    datetime.datetime(2022, 11, 10),
    datetime.datetime(2022, 12, 13),
]
fed_meetings_2022_dates = [
    datetime.datetime(2022, 1, 25),
    datetime.datetime(2022, 1, 26),
    datetime.datetime(2022, 3, 15),
    datetime.datetime(2022, 3, 16),
    datetime.datetime(2022, 4, 26),
    datetime.datetime(2022, 4, 27),
    datetime.datetime(2022, 6, 14),
    datetime.datetime(2022, 6, 15),
    datetime.datetime(2022, 7, 26),
    datetime.datetime(2022, 7, 27),
    datetime.datetime(2022, 9, 20),
    datetime.datetime(2022, 9, 21),
    datetime.datetime(2022, 11, 1),
    datetime.datetime(2022, 11, 2),
    datetime.datetime(2022, 12, 13),
    datetime.datetime(2022, 12, 14),
]

nonfarm_payrolls_2023_dates = [
    datetime.datetime(2023, 3, 3),
    datetime.datetime(2023, 4, 7),
    datetime.datetime(2023, 5, 5),
    datetime.datetime(2023, 6, 2),
    datetime.datetime(2023, 7, 7),
    datetime.datetime(2023, 8, 4),
    datetime.datetime(2023, 9, 1),
    datetime.datetime(2023, 10, 6),
    datetime.datetime(2023, 11, 3),
    datetime.datetime(2023, 12, 1),
    ]

cpi_pce_ppi_2023_dates = [
    datetime.datetime(2023, 2, 14), 
    datetime.datetime(2023, 3, 14),
    datetime.datetime(2023, 4, 12),
    datetime.datetime(2023, 5, 10),
    datetime.datetime(2023, 6, 13),
    datetime.datetime(2023, 7, 12),
    datetime.datetime(2023, 8, 10),
    datetime.datetime(2023, 9, 13),
    datetime.datetime(2023, 10, 12),
    datetime.datetime(2023, 11, 14),
    datetime.datetime(2023, 12, 12),
]
fed_meetings_2023_dates = [
    datetime.datetime(2023, 3, 21),
    datetime.datetime(2023, 3, 22),
    datetime.datetime(2023, 5, 2),
    datetime.datetime(2023, 5, 3),
    datetime.datetime(2023, 6, 6),
    datetime.datetime(2023, 6, 14),
    datetime.datetime(2023, 7, 25),
    datetime.datetime(2023, 7, 26),
    datetime.datetime(2023, 9, 19),
    datetime.datetime(2023, 9, 20),
    datetime.datetime(2023, 10, 31),
    datetime.datetime(2023, 11, 1),
    datetime.datetime(2023, 12, 12),
    datetime.datetime(2023, 12, 13),
]

news_dates = []
news_dates.extend(nonfarm_payrolls_2015_dates)
news_dates.extend(cpi_pce_ppi_2015_dates)
news_dates.extend(fed_meetings_2015_dates)
news_dates.extend(nonfarm_payrolls_2016_dates)
news_dates.extend(cpi_pce_ppi_2016_dates)
news_dates.extend(fed_meetings_2016_dates)
news_dates.extend(nonfarm_payrolls_2017_dates)
news_dates.extend(cpi_pce_ppi_2017_dates)
news_dates.extend(fed_meetings_2017_dates)
news_dates.extend(nonfarm_payrolls_2018_dates)
news_dates.extend(cpi_pce_ppi_2018_dates)
news_dates.extend(fed_meetings_2018_dates)
news_dates.extend(nonfarm_payrolls_2019_dates)
news_dates.extend(cpi_pce_ppi_2019_dates)
news_dates.extend(fed_meetings_2019_dates)
news_dates.extend(nonfarm_payrolls_2020_dates)
news_dates.extend(cpi_pce_ppi_2020_dates)
news_dates.extend(fed_meetings_2020_dates)
news_dates.extend(nonfarm_payrolls_2021_dates)
news_dates.extend(cpi_pce_ppi_2021_dates)
news_dates.extend(fed_meetings_2021_dates)
news_dates.extend(nonfarm_payrolls_2022_dates)
news_dates.extend(cpi_pce_ppi_2022_dates)
news_dates.extend(fed_meetings_2022_dates)
news_dates.extend(nonfarm_payrolls_2023_dates)
news_dates.extend(cpi_pce_ppi_2023_dates)
news_dates.extend(fed_meetings_2023_dates)

def check_weekday():
    good_days = ['Mon', 'Tue', "Wed", 'Thu']
    d = strftime("%a", localtime())
    t = strftime("%H:%M", localtime())
    if d in good_days:
        return True
    if d == 'Fri':
        if t < '16:00':
            return True
        else:
            return False
    if d == "Sat":
        return False
    if d == "Sun":
        if t > '17:00':
            return True
        else:
            return False
        
def check_friday_4pm():
    d = strftime("%a", localtime())
    t = strftime("%H:%M", localtime())
    if d == 'Fri' and t == '16:00':
        return True
    else:
        return False

def check_session(t1: str, t2: str):
    t = strftime("%H:%M", localtime())
    if t > t1 and t < t2:
        return True
    else:
        return False

def check_cpi():
        
    URL = "https://www.bls.gov/schedule/news_release/cpi.htm"
    page = requests.get(URL)

    soup = BeautifulSoup(page.content, "html.parser")

    results = soup.find(id="bodytext")

    subtitles = results.find_all("table", class_="release-list")

    release_data = []
    for table in subtitles:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if cells:
                release_date = cells[1].text
                release_time = cells[2].text
                release_data.append({"date": release_date, "time": release_time})

    for i in release_data:
        date = parser.parse(i['date'])
        release_date = date.date()

        current_date = datetime.now().date()
        if current_date == release_date:
            return True
        else:
            continue
    return False

def calculate_news_impact(acc=None, current_time=None):
    if acc is None:
        current_time = datetime.datetime.now()
    else:
        current_time = current_time.replace(tzinfo=None)

    # Get the most recent and upcoming news dates

    prev_date = max([d for d in news_dates if d < current_time])
    next_date = min([d for d in news_dates if d > current_time])

    total_seconds = (next_date - prev_date).total_seconds()
    elapsed_seconds = (current_time - prev_date).total_seconds()

    # Create the curved line of values
    phase = 0  # set the phase to zero, since we want the cosine wave to start at its maximum value
    amplitude = 1  # set the amplitude to 1
    # set the frequency so that the wave completes one cycle over the total time difference
    frequency = 2 * np.pi / total_seconds
    offset = 0  # set the offset to zero, since we want the wave to start at its maximum value

    current_value = amplitude * np.cos(frequency * elapsed_seconds + phase) + offset

    if current_value > 0:
        return 1, current_value, prev_date, next_date, elapsed_seconds
    else:
        return 2, current_value, prev_date, next_date, elapsed_seconds