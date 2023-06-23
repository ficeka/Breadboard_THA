This project assumes access to environment that can install required dependencies in requirements.txt file

1. Run server.py
2. Navigate to hosted site and upload all csv files within in "data" folder (included)
3. Run Client.py (you should do this while keeping server.py running for the very first run so that client can pull data
 from the server. After the first run (after stocks.db appears in directory) flip the 'run' variable to False to
 prevent downloading data from the server every time)
4. Client.py will pull data from the server api, perform some basic normalization and transformation, and predict the
stock of your choice (via changing the variable 'symbol_to_predict')

5. Thoughts: The overall process is not productionailized. Managed tools like cloud storage and bigqueryML would be a
good solution for this task. No real attempt was made to improve the ML model. The purpose of this file is to
demonstrate use of APIs and data pipelines.