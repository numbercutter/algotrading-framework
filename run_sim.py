from algos.handfruit_prod import handfruit_hedge
from constants import defs
import time
import traceback


if __name__ == "__main__":
    
    buy = handfruit_hedge(symbol="EUR_USD",
                          account_id=defs.MITCH_ACCOUNT_ID,
                          api_k=defs.MITCH_API_KEY,
                          phone_n=defs.MITCH_PHONE,
                          log_m="handfruit_mitch_buy",
                          simulation=True,
                          SELL=True
                          )
    restart = True

    try:
        buy.run(restart=restart)

    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
        restart = True
    time.sleep(60)  # delay of 60 seconds