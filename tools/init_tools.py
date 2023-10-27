import pandas as pd
import constants.defs as defs
from infrastructure.log_wrapper import LogWrapper
from etext import send_sms_via_email
class init_tools:
    def __init__(self, symbol, simulation_data, account_id, error_log, main_log, phone_n):
        self.symbol = symbol
        self.simulation_data = simulation_data
        self.account_id = account_id
        self.error_log = error_log
        self.main_log = main_log
        self.phone_n = phone_n

    def load_data(self, acc, pair, columns=["time", "volume", "mid_c"]):
        """
        load data for every time frame and return dataframes
        """
        

        df_m1 = pd.read_pickle(f"./data/{pair}_M1.pkl")
        df_m5 = pd.read_pickle(f"./data/{pair}_M5.pkl")
        df_m15 = pd.read_pickle(f"./data/{pair}_M15.pkl")
        df_h1 = pd.read_pickle(f"./data/{pair}_H1.pkl")
        df_h4 = pd.read_pickle(f"./data/{pair}_H4.pkl")
        df_d = pd.read_pickle(f"./data/{pair}_D.pkl")

        df_m1 = df_m1[(df_m1.time >= acc.start) & (df_m1.time < acc.end)][columns]
        df_m5 = df_m5[(df_m5.time >= acc.start) & (df_m5.time < acc.end)][columns]
        df_m15 = df_m15[(df_m15.time >= acc.start) & (df_m15.time < acc.end)][columns]
        df_h1 = df_h1[(df_h1.time >= acc.start) & (df_h1.time < acc.end)][columns]
        df_h4 = df_h4[(df_h4.time >= acc.start) & (df_h4.time < acc.end)][columns]
        df_d = df_d[(df_d.time >= acc.start) & (df_d.time < acc.end)][columns]

        df_m1.reset_index(drop=True, inplace=True)
        df_m5.reset_index(drop=True, inplace=True)
        df_m15.reset_index(drop=True, inplace=True)
        df_h1.reset_index(drop=True, inplace=True)
        df_h4.reset_index(drop=True, inplace=True)
        df_d.reset_index(drop=True, inplace=True)

        return df_m1, df_m5, df_m15, df_h1, df_h4, df_d

    def setup_logs(self):
        """
        Set up error and main log files
        """
        self.logs = {}
        self.logs[self.error_log] = LogWrapper(self.error_log)
        self.logs[self.main_log] = LogWrapper(self.main_log)
        self.log_to_main(f"Bot started {self.account_id}")

    def log_message(self, msg, key):
        """
        Log a message to a specified log
        """
        self.logs[key].logger.debug(msg)

    def log_to_main(self, msg):
        """
        Log a message to the main log
        """
        self.log_message(msg, self.main_log)

    def log_to_error(self, msg):
        """
        Log a message to the error log
        """
        self.log_message(msg, self.error_log)

    def phone_alert(self, msg):
        """
        Send a text message as an alert
        """
        # Send a text message using the send_sms_via_email function
        message = send_sms_via_email(
            self.phone_n,
            msg,
            defs.SMS_PROVIDER,
            defs.SMS_CREDENTIALS,
            subject="sent using etext",
        )
        # Log the message and the Twilio SID
        self.log_to_main(f"Phone Alert: {msg}, Twilio sid: {message}")
