import pandas as pd
from flask import Flask, jsonify, request
from distutils.log import debug
from fileinput import filename
from flask import *
import os
from werkzeug.utils import secure_filename
import sqlite3

TEMPLATES_AUTO_RELOAD = True

print('success!')
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
app = Flask(__name__)
# Only allow csv files to be uploaded for this example

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'CHANGE-> use more scure python module to generate if productionalizing'

conn = sqlite3.connect('stocks.db')

conn.execute('CREATE TABLE IF NOT EXISTS stocks (Date TEXT, Open REAL, Symbol TEXT)')
print("Table created successfully")
conn.commit()

table_name = 'stocks'

@app.route('/', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
        upload_files = request.files.getlist('file')
        with sqlite3.connect("stocks.db") as conn:
            for f in upload_files:
                df = pd.read_csv(f, index_col=[0])
                df.to_sql(table_name, conn, if_exists='append', index=False)
            conn.commit()
        return render_template('index_posted.html')
    return render_template("index.html")


@app.route('/<date>', methods=['GET'])
def downloadFile(date):
    if request.method == 'GET':
        with sqlite3.connect("stocks.db") as conn:
            # cur = conn.cursor()
            sql_query = pd.read_sql_query(
                '''
                SELECT *
                FROM stocks
                ''',
                conn)
            df = pd.DataFrame(sql_query)
            data = df.to_dict()
        return {'data': data}, 200

if __name__ == '__main__':
 app.run(debug=True, host='0.0.0.0', port=9020)