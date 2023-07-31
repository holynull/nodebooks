CMC_QUOTE_LASTEST_API_DOC="""
Quotes Latest v2 API Documentation
Base url is https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest?convert=USD
Returns the latest market quote for 1 or more cryptocurrencies. Use the "convert" option to return market values in multiple fiat and cryptocurrency conversions in the same call.
There is no need to use aux to specify a specific market data, and the returned quote contains all market data.
PARAMETERS
slug is Alternatively pass a comma-separated list of cryptocurrency slugs. Example: "bitcoin,ethereum"
symbol is Alternatively pass one or more comma-separated cryptocurrency symbols. Example: "BTC,ETH". At least one "id" or "slug" or "symbol" is required for this request.
convert is Optionally calculate market quotes in up to 120 currencies at once by passing a comma-separated list of cryptocurrency or fiat currency symbols. Each additional convert option beyond the first requires an additional call credit. A list of supported fiat options can be found here. Each conversion is returned in its own "quote" object.
The responsing's explaination as following as following.
RESPONSE
id is The unique CoinMarketCap ID for this cryptocurrency.
name is The name of this cryptocurrency.
symbol is The ticker symbol for this cryptocurrency.
slug is The web URL friendly shorthand version of this cryptocurrency name.
cmc_rank is The cryptocurrency's CoinMarketCap rank by market cap.
num_market_pairs is The number of active trading pairs available for this cryptocurrency across supported exchanges.
circulating_supply is The approximate number of coins circulating for this cryptocurrency.
total_supply is The approximate total amount of coins in existence right now (minus any coins that have been verifiably burned).
market_cap_by_total_supply is The market cap by total supply. This field is only returned if requested through the aux request parameter.
max_supply is The expected maximum limit of coins ever to be available for this cryptocurrency.
date_added is Timestamp (ISO 8601) of when this cryptocurrency was added to CoinMarketCap.
tags is Array of tags associated with this cryptocurrency. Currently only a mineable tag will be returned if the cryptocurrency is mineable. Additional tags will be returned in the future.
platform is Metadata about the parent cryptocurrency platform this cryptocurrency belongs to if it is a token, otherwise null.
self_reported_circulating_supply is The self reported number of coins circulating for this cryptocurrency.
self_reported_market_cap is The self reported market cap for this cryptocurrency.
quote is A map of market quotes in different currency conversions. The default map included is USD. See the flow Quote Map Instructions.
Quote Map Instructions
price is Price in the specified currency.
volume_24h is Rolling 24 hour adjusted volume in the specified currency.
volume_change_24h is 24 hour change in the specified currencies volume.
volume_24h_reported is Rolling 24 hour reported volume in the specified currency. This field is only returned if requested through the aux request parameter.
volume_7d is Rolling 7 day adjusted volume in the specified currency. This field is only returned if requested through the aux request parameter.
volume_7d_reported is Rolling 7 day reported volume in the specified currency. This field is only returned if requested through the aux request parameter.
volume_30d is Rolling 30 day adjusted volume in the specified currency. This field is only returned if requested through the aux request parameter.
volume_30d_reported is Rolling 30 day reported volume in the specified currency. This field is only returned if requested through the aux request parameter.
market_cap is Market cap in the specified currency.
market_cap_dominance is Market cap dominance in the specified currency.
fully_diluted_market_cap is Fully diluted market cap in the specified currency.
percent_change_1h is Percentage price increase within 1 hour in the specified currency.
percent_change_24h is Percentage price increase within 24 hour in the specified currency.
percent_change_7d is Percentage price increase within 7 day in the specified currency.
percent_change_30d is Percentage price increase within 30 day in the specified currency.
"""