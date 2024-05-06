# 36c17c6e547713acd4e687d42d19affe47bc4bf7cc94fa4259a57e8ec53d4ce1

# from sec_api import QueryApi

# queryApi = QueryApi(api_key="36c17c6e547713acd4e687d42d19affe47bc4bf7cc94fa4259a57e8ec53d4ce1")

# query = {
#   "query": "ticker:MSFT AND filedAt:[1995-01-01 TO 2023-12-31] AND formType:\"10-K\"",
#   "from": "0",
#   "size": "29",
#   "sort": [{ "filedAt": { "order": "desc" } }]
# }

# filings = queryApi.get_filings(query)

# print(filings)


# from sec_api import XbrlApi

# xbrlApi = XbrlApi("36c17c6e547713acd4e687d42d19affe47bc4bf7cc94fa4259a57e8ec53d4ce1")

# # 10-K HTM File URL example
# xbrl_json = xbrlApi.xbrl_to_json(
#     htm_url="https://www.sec.gov/Archives/edgar/data/320193/000032019320000096/aapl-20200926.htm"
# )

# # access income statement, balance sheet and cash flow statement
# print(xbrl_json["StatementsOfIncome"])
# print(xbrl_json["BalanceSheets"])
# print(xbrl_json["StatementsOfCashFlows"])

# # 10-K XBRL File URL example
# xbrl_json = xbrlApi.xbrl_to_json(
#     xbrl_url="https://www.sec.gov/Archives/edgar/data/1318605/000156459021004599/tsla-10k_20201231_htm.xml"
# )

# # 10-K accession number example
# xbrl_json = xbrlApi.xbrl_to_json(accession_no="0001564590-21-004599")

from sec_edgar_downloader import Downloader

dl = Downloader("IIIT", "devesh.marwah@research.iiit.ac.in")

# dl.get("10-K", "TSLA", limit=1)
dl.get("10-K", "AAPL", after="1995-01-01", before="2024-01-01")