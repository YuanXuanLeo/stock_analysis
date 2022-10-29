from turtle import title
from predict.model import perform_training
from flask import Flask,render_template,request
import app.run as run
import app.app as app
import utils
import json
import os

from flask import Flask,render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)


@app.route('/',methods=['GET'])
def index():
    active = 'active'
    title = 'US-TOCK'
    return render_template('index.html', active = active, title= title)

@app.route('/index_sp500.html')
def index_sp500():
	title = "即時股價"
	return render_template('index_sp500.html', title=title)

@app.route("/index_dow.html")
def index_dow():
    title = "即時股價"
    return render_template("index_dow.html", title=title)

@app.route("/index_nasdaq100.html")
def index_nasdaq100():
    title = "即時股價"
    return render_template("index_nasdaq100.html", title=title)


@app.route('/dowJones')
def dowJones():
    f=open('./static/historical_DowJones.json')
    data=json.load(f)
    f.close()
    return {"res":data}

@app.route('/nasdaq')
def nasdaq():
    f=open('./static/historical_nasdaq100.json')
    data=json.load(f)
    f.close()
    return {"res":data}

@app.route('/sp500')
def sp500():
    f=open('./static/new_historical_sp500.json')
    data=json.load(f)
    f.close()
    return {"res":data}

@app.route('/usa_indices')
def usaIndices():
    f=open('./static/USA_indices_history.json')
    data=json.load(f)
    f.close()
    return {"res":data}

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stock_list.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
with app.app_context():

    class all_stock(db.Model):
        marketindexes = db.Column(db.String(30))
        symbol = db.Column(db.String(30), primary_key=True,unique = False)
        previousclose = db.Column(db.Integer, index=True,nullable=True)
        cheapprice = db.Column(db.Integer, index=True,nullable=True)
        fairprice = db.Column(db.Integer, index=True,nullable=True)
        expensiveprice = db.Column(db.Integer, index=True,nullable=True)
        volume = db.Column(db.Integer, index=True,nullable=True)
        dividendyield = db.Column(db.Integer, index=True,nullable=True)
        fiveyearavgdividendyield = db.Column(db.Integer, index=True,nullable=True)
        lastdividendvalue = db.Column(db.Integer, index=True,nullable=True)

        def to_dict1(self):
            return {
                'marketindexes': self.marketindexes,
                'symbol': self.symbol,
                'previousclose': self.previousclose,
                'cheapprice': self.cheapprice,
                'fairprice': self.fairprice,
                'expensiveprice': self.expensiveprice,
                'volume': self.volume,
                'dividendyield': self.dividendyield,
                'fiveyearavgdividendyield': self.fiveyearavgdividendyield,
                'lastdividendvalue': self.lastdividendvalue
            }

    class dji(db.Model):
        marketindexes = db.Column(db.String(30))
        symbol = db.Column(db.String(30), primary_key=True,unique = False)
        previousclose = db.Column(db.Integer, index=True,nullable=True)
        cheapprice = db.Column(db.Integer, index=True,nullable=True)
        fairprice = db.Column(db.Integer, index=True,nullable=True)
        expensiveprice = db.Column(db.Integer, index=True,nullable=True)
        volume = db.Column(db.Integer, index=True,nullable=True)
        dividendyield = db.Column(db.Integer, index=True,nullable=True)
        fiveyearavgdividendyield = db.Column(db.Integer, index=True,nullable=True)
        lastdividendvalue = db.Column(db.Integer, index=True,nullable=True)

        def to_dict2(self):
            return {
                'marketindexes': self.marketindexes,
                'symbol': self.symbol,
                'previousclose': self.previousclose,
                'cheapprice': self.cheapprice,
                'fairprice': self.fairprice,
                'expensiveprice': self.expensiveprice,
                'volume': self.volume,
                'dividendyield': self.dividendyield,
                'fiveyearavgdividendyield': self.fiveyearavgdividendyield,
                'lastdividendvalue': self.lastdividendvalue
            }

    class sp500(db.Model):
        marketindexes = db.Column(db.String(30))
        symbol = db.Column(db.String(30), primary_key=True,unique = False)
        previousclose = db.Column(db.Integer, index=True,nullable=True)
        cheapprice = db.Column(db.Integer, index=True,nullable=True)
        fairprice = db.Column(db.Integer, index=True,nullable=True)
        expensiveprice = db.Column(db.Integer, index=True,nullable=True)
        volume = db.Column(db.Integer, index=True,nullable=True)
        dividendyield = db.Column(db.Integer, index=True,nullable=True)
        fiveyearavgdividendyield = db.Column(db.Integer, index=True,nullable=True)
        lastdividendvalue = db.Column(db.Integer, index=True,nullable=True)

        def to_dict3(self):
            return {
                'marketindexes': self.marketindexes,
                'symbol': self.symbol,
                'previousclose': self.previousclose,
                'cheapprice': self.cheapprice,
                'fairprice': self.fairprice,
                'expensiveprice': self.expensiveprice,
                'volume': self.volume,
                'dividendyield': self.dividendyield,
                'fiveyearavgdividendyield': self.fiveyearavgdividendyield,
                'lastdividendvalue': self.lastdividendvalue
            }

    class nasdaq(db.Model):
        marketindexes = db.Column(db.String(30))
        symbol = db.Column(db.String(30), primary_key=True,unique = False)
        previousclose = db.Column(db.Integer, index=True,nullable=True)
        cheapprice = db.Column(db.Integer, index=True,nullable=True)
        fairprice = db.Column(db.Integer, index=True,nullable=True)
        expensiveprice = db.Column(db.Integer, index=True,nullable=True)
        volume = db.Column(db.Integer, index=True,nullable=True)
        dividendyield = db.Column(db.Integer, index=True,nullable=True)
        fiveyearavgdividendyield = db.Column(db.Integer, index=True,nullable=True)
        lastdividendvalue = db.Column(db.Integer, index=True,nullable=True)

        def to_dict4(self):
            return {
                'marketindexes': self.marketindexes,
                'symbol': self.symbol,
                'previousclose': self.previousclose,
                'cheapprice': self.cheapprice,
                'fairprice': self.fairprice,
                'expensiveprice': self.expensiveprice,
                'volume': self.volume,
                'dividendyield': self.dividendyield,
                'fiveyearavgdividendyield': self.fiveyearavgdividendyield,
                'lastdividendvalue': self.lastdividendvalue
            }

    db.create_all()
    

						   
@app.route('/risk.html')
def search():
	title = "股票篩選"
	return render_template('risk.html', title=title)

@app.route('/stocktable/all')
def alls():
    return {'data': [user.to_dict1() for user in all_stock.query]}
@app.route('/stocktable/dji')
def djis():
    return {'data': [user.to_dict2() for user in dji.query]}
@app.route('/stocktable/sp500')
def sp500s():
    return {'data': [user.to_dict3() for user in sp500.query]}
@app.route('/stocktable/nasdaq')
def nasdaqs():
    return {'data': [user.to_dict4() for user in nasdaq.query]}    



# @app.route('/predict.html')
# def charts():
# 	title = "預測模型"
# 	return render_template('predict.html', title=title)

all_files = utils.read_all_stock_files('predict/individual_stocks_5yr')
@app.route('/predict.html')
def landing_function():
    title = "股價預測"
    # df = all_files['A']
    # df = pd.read_csv('GOOG_30_days.csv')
    # all_prediction_data, all_prediction_data, prediction_date, dates, all_data, all_data = perform_training('A', df, ['SVR_linear'])
    stock_files = list(all_files.keys())
    return render_template('predict.html',show_results="false", stocklen=len(stock_files), stock_files=stock_files, len2=len([]),
                           all_prediction_data=[], title=title,
                           prediction_date="", dates=[], all_data=[], len=len([]))

@app.route('/process', methods=['POST'])
def process():
    title = "股價預測"
    stock_file_name = request.form['stockfile']
    ml_algoritms = request.form.getlist('mlalgos')
    df = all_files[str(stock_file_name)]
    stockname = str(stock_file_name)
    all_prediction_data, all_prediction_data, prediction_date, dates, all_data, all_data, all_test_evaluations = \
        perform_training(str(stock_file_name), df, ml_algoritms)
    stock_files = list(all_files.keys())

    return render_template('predict.html', all_test_evaluations=all_test_evaluations, show_results="true",
                           stocklen=len(stock_files), stock_files=stock_files,
                           len2=len(all_prediction_data),
                           all_prediction_data=all_prediction_data,
                           prediction_date=prediction_date, dates=dates, all_data=all_data, len=len(all_data),
                           stockname=stockname)

@app.route('/index_treemap.html')
def treemap():
	title = "大盤板塊"
	return render_template('index_treemap.html', title=title)

@app.route("/<market>/<day>",methods=["GET"])
def getTreemap(market,day):
    return render_template(f"{market}_{day}.html")

@app.route('/dowJones')
def dowJones_json():
    f=open('./static/historical_DowJones.json')
    data=json.load(f)
    f.close()
    return {"res":data}

@app.route('/nasdaq')
def nasdaq_json():
    f=open('./static/historical_nasdaq100.json')
    data=json.load(f)
    f.close()
    return {"res":data}

@app.route('/sp500')
def sp500_json():
    f=open('./static/new_historical_sp500.json')
    print(f)
    data=json.load(f)
    print(data)
    f.close()
    return {"res":data}

@app.route('/dowJones_list')
def dowJones_list():
    f=open('./static/DowJones_list.json')
    data=json.load(f)
    f.close()
    return {"res":data}

@app.route('/nasdaq_list')
def nasdaq_list():
    f=open('./static/Nasdaq_list.json')
    data=json.load(f)
    f.close()
    return {"res":data}

@app.route('/sp500_list')
def sp500_list():
    f=open('./static/SP500_list.json')
    print(f)
    data=json.load(f)
    print(data)
    f.close()
    return {"res":data}

@app.route('/usa_indices')
def usaIndices_json():
    f=open('./static/USA_indices_history.json')
    data=json.load(f)
    f.close()
    return {"res":data}


@app.route('/news.html')
def analytics():
	title = "國際新聞"
	return render_template('news.html', title=title)

@app.route('/index.html')
def team():
	title = "團隊介紹"
	return render_template('index.html',title=title)

if __name__=="__main__":
	app.run(debug=True, port=5001)