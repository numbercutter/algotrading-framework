from api.oanda_api_sim import OandaApiSim
from constants import defs
import schedule
import time

def generate_summary(api, account_name, file):
    ids = api.get_account_ids()
    fields = ['ID', 'Created Time', 'Currency', 'Alias', 'Balance', 'PL', 'Margin Rate', 'Open Trade Count', 'Open Position Count', 'Pending Order Count', 'NAV', 'Margin Used', 'Margin Available', 'Withdrawal Limit']
    summary_totals = {field: 0 for field in fields[4:]}
    data_rows = []
    column_widths = [len(field) for field in fields]

    for id in ids:
        summary = api.get_account_summary(id)
        row = [
            id,
            summary['createdTime'],
            summary['currency'],
            summary['alias'],
            summary['balance'],
            summary['pl'],
            summary['marginRate'],
            summary['openTradeCount'],
            summary['openPositionCount'],
            summary['pendingOrderCount'],
            summary['NAV'],
            summary['marginUsed'],
            summary['marginAvailable'],
            summary['withdrawalLimit'],
        ]
        data_rows.append(row)
        column_widths = [max(width, len(str(value))) for width, value in zip(column_widths, row)]

        # Add values to summary totals
        for i, field in enumerate(fields[4:]):
            summary_totals[field] += float(row[i + 4])

    # Write the header
    header = "".join([f"{field:<{width+2}}" for field, width in zip(fields, column_widths)])
    file.write(account_name + "\n" + header + "\n")
    file.write("=" * len(header) + "\n")

    for row in data_rows:
        row_str = "".join([f"{str(value):<{width+2}}" for value, width in zip(row, column_widths)])
        file.write(row_str + "\n")

    # Write a separator and summary table
    file.write("\n" + "=" * 50 + "\nSummary for " + account_name + ":\n" + "=" * 50 + "\n")
    for field, value in summary_totals.items():
        file.write(f"{field:<30} {value}\n")
    file.write("\n" * 2)

def generate_report():
    mitch_api = OandaApiSim(api_key=defs.MITCH_API_KEY, oanda_url=defs.OANDA_URL, account_id='001-001-1882287-011')
    
    with open('accounts_report.txt', mode='w') as file:
        generate_summary(mitch_api, "Mitch's Account", file)

    print("Report generated successfully in accounts_report.txt")


# Generate the report once when the script is first run
generate_report()

# Keep the script running
while True:
    time.sleep(60 * 60 * 24)  # Sleep for 24 hours
    generate_report()

