import pandas as pd
from flask import Flask, request, render_template
from ml_engg import predictions_2022_df

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    date=None
    if request.method == 'POST':
        date = request.form['date']
        Date = request.form['date']
        Date = str(Date)
        row_position = predictions_2022_df.index.get_loc(Date)
        value = predictions_2022_df.iloc[row_position]
        result = int(value)
    return render_template('index.html', result=result, date=date)



if __name__ == '__main__':
    app.run(debug=True)