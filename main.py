from api.oanda_api_sim import OandaApiSim
from infrastructure.instrument_collection import instrumentCollection
from dateutil import parser
from infrastructure.collect_data import run_collection
from constants import defs

if __name__ == '__main__':
    api = OandaApiSim(
            api_key=defs.API_KEY,
            oanda_url=defs.OANDA_URL,
            account_id=defs.HR_ACCOUNT_ID
            )
    instrumentCollection.LoadInstruments("./data")
    run_collection(instrumentCollection, api)
    #run_ema_macd(instrumentCollection)
    #run_ema_macd(instrumentCollection)
    