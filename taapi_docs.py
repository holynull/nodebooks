RSI_API_DOCS="""Relative Strength Index (RSI)
Base URL is https://api.taapi.io/rsi
The relative strength index (RSI) is a momentum indicator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset. The RSI is displayed as an oscillator (a line graph that moves between two extremes) and can have a reading from 0 to 100.
The RSI provides signals that tell investors to buy when the security or currency is oversold and to sell when it is overbought. 
API PARAMETERS
secret is the secret which is emailed to you when you request an API key. 
Note that the secret is: {taapi_key} 
exchange is the exchange you want to calculate the indicator from: gateio or one of our supported exchanges. For other crypto / stock exchanges, please refer to our Client or Manual integration methods.
symbol is default means the coin to USDT. It is always uppercase, with the coin separated by a forward slash and USDT: COIN/USDT. For example: BTC/USDT Bitcoin to Tether, or LTC/USDT Litecoin to Tether...
interval is interval or time frame: We support the following time frames: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d, 1w. So if you're interested in values on hourly candles, use interval=1h, for daily values use interval=1d, etc.
MORE EXAMPLES
Let's say you want to know the rsi value on the last closed candle on the 30m timeframe. You are not interest in the real-time value, so you use the backtrack=1 optional parameter to go back 1 candle in history to the last closed candle.
```
https://api.taapi.io/rsi?secret={taapi_key}&exchange=gateio&symbol=BTC/USDT&interval=30m&backtrack=1
```
Get rsi values on each of the past X candles in one call:
Let's say you want to know what the rsi daily value was each day for the previous 10 days. You can get this returned by our API easily and efficiently in one call using the backtracks=10 parameter:
```
https://api.taapi.io/rsi?secret={taapi_key}&exchange=gateio&symbol=BTC/USDT&interval=1d&backtracks=10
```
Note that parameter symbol must be changed to cryptocurrency symbol separated by a forward slash and USDT, like BTC/USDT.
"""

CCI_API_DOCS="""Commodity Channel Index
Base URL is https://api.taapi.io/cci
Developed by Donald Lambert, the Commodity Channel Index​ (CCI) is a momentum-based oscillator used to help determine when an asset is reaching overbought or oversold conditions.
API PARAMETERS
secret is The secret which is emailed to you when you request an API key. 
Note that The secret is: {taapi_key} 
exchange is The exchange you want to calculate the indicator from: gateio or one of our supported exchanges. For other crypto / stock exchanges, please refer to our Client or Manual integration methods.
symbol is default means the coin to USDT. It is always uppercase, with the coin separated by a forward slash and USDT: COIN/USDT. For example: BTC/USDT Bitcoin to Tether, or LTC/USDT Litecoin to Tether...
interval is Interval or time frame: We support the following time frames: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d, 1w. So if you're interested in values on hourly candles, use interval=1h, for daily values use interval=1d, etc.
MORE EXAMPLES
Let's say you want to know the cci value on the last closed candle on the 30m timeframe. You are not interest in the real-time value, so you use the backtrack=1 optional parameter to go back 1 candle in history to the last closed candle.
```
https://api.taapi.io/cci?secret={taapi_key} &exchange=gateio&symbol=BTC/USDT&interval=30m&backtrack=1
```
Get cci values on each of the past X candles in one call
Let's say you want to know what the cci daily value was each day for the previous 10 days. You can get this returned by our API easily and efficiently in one call using the backtracks=10 parameter:
```
https://api.taapi.io/cci?secret={taapi_key} &exchange=gateio&symbol=BTC/USDT&interval=1d&backtracks=10
```
Note that parameter symbol must be changed to cryptocurrency symbol separated by a forward slash and USDT, like BTC/USDT.
"""

DMI_API_DOCS="""Directional Movement Index
Base URL is https://api.taapi.io/dmi 
The dmi endpoint returns a JSON response like this
```json
{{
  "adx": 40.50793463106886,
  "plusdi": 33.32334015840893,
  "minusdi": 10.438557555722891
}}
```
API PARAMETERS
secret is The secret which is emailed to you when you request an API key. 
Note that The secret is: {taapi_key} 
exchange is The exchange you want to calculate the indicator from: gateio or one of our supported exchanges. For other crypto / stock exchanges, please refer to our Client or Manual integration methods.
symbol is default means the coin to USDT. It is always uppercase, with the coin separated by a forward slash and USDT: COIN/USDT. For example: BTC/USDT Bitcoin to Tether, or LTC/USDT Litecoin to Tether...
interval is Interval or time frame: We support the following time frames: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d, 1w. So if you're interested in values on hourly candles, use interval=1h, for daily values use interval=1d, etc.
More examples
Let's say you want to know the dmi value on the last closed candle on the 30m timeframe. You are not interest in the real-time value, so you use the backtrack=1 optional parameter to go back 1 candle in history to the last closed candle.
```
https://api.taapi.io/dmi?secret={taapi_key}&exchange=gateio&symbol=BTC/USDT&interval=30m&backtrack=1
```
Get dmi values on each of the past X candles in one call
Let's say you want to know what the dmi daily value was each day for the previous 10 days. You can get this returned by our API easily and efficiently in one call using the backtracks=10 parameter:
```
https://api.taapi.io/dmi?secret={taapi_key}&exchange=gateio&symbol=BTC/USDT&interval=1d&backtracks=10
```
Note that parameter symbol must be changed to cryptocurrency symbol separated by a forward slash and USDT, like BTC/USDT.
"""

MACD_API_DOCS="""Moving Average Convergence Divergence (MACD)
Base URL is https://api.taapi.io/macd 
The dmi endpoint returns a JSON response like this
```json
{{
  "valueMACD": 737.4052287912818,
  "valueMACDSignal": 691.8373005221695,
  "valueMACDHist": 45.56792826911237
}}
```
API PARAMETERS
secret is The secret which is emailed to you when you request an API key. 
Note that The secret is: {taapi_key} 
exchange is The exchange you want to calculate the indicator from: gateio or one of our supported exchanges. For other crypto / stock exchanges, please refer to our Client or Manual integration methods.
symbol is default means the coin to USDT. It is always uppercase, with the coin separated by a forward slash and USDT: COIN/USDT. For example: BTC/USDT Bitcoin to Tether, or LTC/USDT Litecoin to Tether...
interval is Interval or time frame: We support the following time frames: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d, 1w. So if you're interested in values on hourly candles, use interval=1h, for daily values use interval=1d, etc.
MORE EXAMPLES
Let's say you want to know the macd value on the last closed candle on the 30m timeframe. You are not interest in the real-time value, so you use the backtrack=1 optional parameter to go back 1 candle in history to the last closed candle.
```
https://api.taapi.io/macd?secret={taapi_key}&exchange=gateio&symbol=BTC/USDT&interval=30m&backtrack=1
```
Get macd values on each of the past X candles in one call
Let's say you want to know what the macd daily value was each day for the previous 10 days. You can get this returned by our API easily and efficiently in one call using the backtracks=10 parameter:
```
https://api.taapi.io/macd?secret={taapi_key}&exchange=gateio&symbol=BTC/USDT&interval=1d&backtracks=10
```
Note that parameter symbol must be changed to cryptocurrency symbol separated by a forward slash and USDT, like BTC/USDT.
"""

PSAR_API_DOCS="""Parabolic SAR
Base URL is https://api.taapi.io/psar 
The Parabolic SAR (Stop and Reverse) is a popular technical indicator used in financial markets to identify potential trend reversals. Developed by J. Welles Wilder Jr., it utilizes a series of dots plotted on a price chart to highlight potential entry and exit points.
The Parabolic SAR dots are positioned above or below the price, depending on the direction of the prevailing trend. When the dots are below the price, it indicates an uptrend, while dots above the price indicate a downtrend. The dots gradually adjust their position as the price evolves, creating a parabolic shape. Traders often use the Parabolic SAR as a tool for setting stop-loss orders and trailing stops, as it dynamically adapts to market conditions and can help identify potential trend reversals in a timely manner.
In the Parabolic SAR formula, users have the flexibility to adjust three optional parameters: start, increment and maximum. The start parameter determines the initial acceleration factor for the indicator. The increment parameter specifies the rate at which the acceleration factor increases over time. Lastly, the maximum parameter sets the upper limit for the acceleration factor. By modifying these parameters, traders can customize the sensitivity and responsiveness of the Parabolic SAR indicator to suit their trading strategies and market conditions.
API PARAMETERS
secret is The secret which is emailed to you when you request an API key. 
Note that The secret is {taapi_key} 
exchange is The exchange you want to calculate the indicator from gateio or one of our supported exchanges. For other crypto / stock exchanges, please refer to our Client or Manual integration methods.
symbol is default means the coin to USDT. It is always uppercase, with the coin separated by a forward slash and USDT: COIN/USDT. For example: BTC/USDT Bitcoin to Tether, or LTC/USDT Litecoin to Tether...
interval is Interval or time frame: We support the following time frames: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d, 1w. So if you're interested in values on hourly candles, use interval=1h, for daily values use interval=1d, etc.
MORE EXAMPLES
Let's say you want to know the psar value on the last closed candle on the 30m timeframe. You are not interest in the real-time value, so you use the backtrack=1 optional parameter to go back 1 candle in history to the last closed candle.
```
https://api.taapi.io/psar?secret={taapi_key}&exchange=gateio&symbol=BTC/USDT&interval=30m&backtrack=1
```
Get psar values on each of the past X candles in one call
Let's say you want to know what the psar daily value was each day for the previous 10 days. You can get this returned by our API easily and efficiently in one call using the backtracks=10 parameter:
```
https://api.taapi.io/psar?secret={taapi_key}&exchange=gateio&symbol=BTC/USDT&interval=1d&backtracks=10
```
Note that parameter symbol must be changed to cryptocurrency symbol separated by a forward slash and USDT, like BTC/USDT.
"""

STOCHRSI_API_DOCS="""Stochastic Relative Strength Index
Base URL is https://api.taapi.io/stochrsi 
The stochrsi endpoint returns a JSON response like this:
{{
	"valueFastK": 28.157463220370534,
  	"valueFastD": 52.369851780456976
}}
API PARAMETERS
secret is The secret which is emailed to you when you request an API key. 
Note that The secret is: {taapi_key} 
exchange is The exchange you want to calculate the indicator from: gateio or one of our supported exchanges. For other crypto / stock exchanges, please refer to our Client or Manual integration methods.
symbol is default means the coin to USDT. It is always uppercase, with the coin separated by a forward slash and USDT: COIN/USDT. For example: BTC/USDT Bitcoin to Tether, or LTC/USDT Litecoin to Tether...
interval is Interval or time frame: We support the following time frames: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d, 1w. So if you're interested in values on hourly candles, use interval=1h, for daily values use interval=1d, etc.
MORE EXAMPLES
Let's say you want to know the stochrsi value on the last closed candle on the 30m timeframe. You are not interest in the real-time value, so you use the backtrack=1 optional parameter to go back 1 candle in history to the last closed candle.
```
https://api.taapi.io/stochrsi?secret={taapi_key}&exchange=gateio&symbol=BTC/USDT&interval=30m&backtrack=1
```
Get stochrsi values on each of the past X candles in one call
Let's say you want to know what the stochrsi daily value was each day for the previous 10 days. You can get this returned by our API easily and efficiently in one call using the backtracks=10 parameter:
```
https://api.taapi.io/stochrsi?secret={taapi_key}&exchange=gateio&symbol=BTC/USDT&interval=1d&backtracks=10
```
Note that parameter symbol must be changed to cryptocurrency symbol separated by a forward slash and USDT, like BTC/USDT.
"""

CMF_API_DOCS="""Chaikin Money Flow
Base URL is https://api.taapi.io/cmf 
API PARAMETERS
secret is The secret which is emailed to you when you request an API key. 
Note that The secret is: {taapi_key} 
exchange is The exchange you want to calculate the indicator from: gateio or one of our supported exchanges. For other crypto / stock exchanges, please refer to our Client or Manual integration methods.
symbol is default means the coin to USDT. It is always uppercase, with the coin separated by a forward slash and USDT: COIN/USDT. For example: BTC/USDT Bitcoin to Tether, or LTC/USDT Litecoin to Tether...
interval is Interval or time frame: We support the following time frames: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d, 1w. So if you're interested in values on hourly candles, use interval=1h, for daily values use interval=1d, etc.
MORE EXAMPLES
Let's say you want to know the cmf value on the last closed candle on the 30m timeframe. You are not interest in the real-time value, so you use the backtrack=1 optional parameter to go back 1 candle in history to the last closed candle.
```
https://api.taapi.io/cmf?secret={taapi_key}&exchange=gateio&symbol=BTC/USDT&interval=30m&backtrack=1
```
Get cmf values on each of the past X candles in one call
Let's say you want to know what the cmf daily value was each day for the previous 10 days. You can get this returned by our API easily and efficiently in one call using the backtracks=10 parameter:
```
https://api.taapi.io/cmf?secret={taapi_key}&exchange=gateio&symbol=BTC/USDT&interval=1d&backtracks=10
```
Note that parameter symbol must be changed to cryptocurrency symbol separated by a forward slash and USDT, like BTC/USDT.
"""