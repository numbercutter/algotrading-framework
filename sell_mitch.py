from algos.handfruit_hedge import handfruit_hedge
from constants import defs
import time
import traceback


if __name__ == "__main__":
    
    sell = handfruit_hedge(symbol="EUR_USD",
                           account_id=defs.MITCH_ACCOUNT_ID,
                           api_k=defs.MITCH_API_KEY,
                           phone_n=defs.MITCH_PHONE,
                           log_m="handfruit_mitch_sell",
                           simulation=False,
                           SELL=True
                           )
    restart = True
    while True:
        try:
            sell.run(restart=restart)
            restart = False
        except Exception as e:
            print(f"Error occurred: {e}")
            traceback.print_exc()
            restart = True
        time.sleep(60)  # delay of 60 seconds