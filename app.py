from turtle import title
from predict.model import perform_training
from flask import Flask,render_template,request
from model_vr1 import random_forests
from model_vr1 import KNN
import app.run as run
import app.app as app
import pandas as pd
import numpy as np
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
    f=open('./static/historical_sp500.json')
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
        close = db.Column(db.Integer, index=True,nullable=True)
        cheapprice = db.Column(db.Integer, index=True,nullable=True)
        fairprice = db.Column(db.Integer, index=True,nullable=True)
        expensiveprice = db.Column(db.Integer, index=True,nullable=True)
        volume = db.Column(db.Integer, index=True,nullable=True)
        dividendyield = db.Column(db.Integer, index=True,nullable=True)
        fiveyearavgdividendyield = db.Column(db.Integer, index=True,nullable=True)
        fiveyeardividend = db.Column(db.Integer, index=True,nullable=True)
        update = db.Column(db.Date, index=True,nullable=True)

        def to_dict1(self):
            return {
                'marketindexes': self.marketindexes,
                'symbol': self.symbol,
                'close': self.close,
                'cheapprice': self.cheapprice,
                'fairprice': self.fairprice,
                'expensiveprice': self.expensiveprice,
                'volume': self.volume,
                'dividendyield': self.dividendyield,
                'fiveyearavgdividendyield': self.fiveyearavgdividendyield,
                'fiveyeardividend': self.fiveyeardividend,
                'update': self.update
            }

    class dji(db.Model):
        marketindexes = db.Column(db.String(30))
        symbol = db.Column(db.String(30), primary_key=True,unique = False)
        close = db.Column(db.Integer, index=True,nullable=True)
        cheapprice = db.Column(db.Integer, index=True,nullable=True)
        fairprice = db.Column(db.Integer, index=True,nullable=True)
        expensiveprice = db.Column(db.Integer, index=True,nullable=True)
        volume = db.Column(db.Integer, index=True,nullable=True)
        dividendyield = db.Column(db.Integer, index=True,nullable=True)
        fiveyearavgdividendyield = db.Column(db.Integer, index=True,nullable=True)
        fiveyeardividend = db.Column(db.Integer, index=True,nullable=True)
        update = db.Column(db.Date, index=True,nullable=True)

        def to_dict2(self):
            return {
                'marketindexes': self.marketindexes,
                'symbol': self.symbol,
                'close': self.close,
                'cheapprice': self.cheapprice,
                'fairprice': self.fairprice,
                'expensiveprice': self.expensiveprice,
                'volume': self.volume,
                'dividendyield': self.dividendyield,
                'fiveyearavgdividendyield': self.fiveyearavgdividendyield,
                'fiveyeardividend': self.fiveyeardividend,
                'update': self.update
            }

    class sp500(db.Model):
        marketindexes = db.Column(db.String(30))
        symbol = db.Column(db.String(30), primary_key=True,unique = False)
        close = db.Column(db.Integer, index=True,nullable=True)
        cheapprice = db.Column(db.Integer, index=True,nullable=True)
        fairprice = db.Column(db.Integer, index=True,nullable=True)
        expensiveprice = db.Column(db.Integer, index=True,nullable=True)
        volume = db.Column(db.Integer, index=True,nullable=True)
        dividendyield = db.Column(db.Integer, index=True,nullable=True)
        fiveyearavgdividendyield = db.Column(db.Integer, index=True,nullable=True)
        fiveyeardividend = db.Column(db.Integer, index=True,nullable=True)
        update = db.Column(db.Date, index=True,nullable=True)

        def to_dict3(self):
            return {
                'marketindexes': self.marketindexes,
                'symbol': self.symbol,
                'close': self.close,
                'cheapprice': self.cheapprice,
                'fairprice': self.fairprice,
                'expensiveprice': self.expensiveprice,
                'volume': self.volume,
                'dividendyield': self.dividendyield,
                'fiveyearavgdividendyield': self.fiveyearavgdividendyield,
                'fiveyeardividend': self.fiveyeardividend,
                'update': self.update
            }

    class nasdaq(db.Model):
        marketindexes = db.Column(db.String(30))
        symbol = db.Column(db.String(30), primary_key=True,unique = False)
        close = db.Column(db.Integer, index=True,nullable=True)
        cheapprice = db.Column(db.Integer, index=True,nullable=True)
        fairprice = db.Column(db.Integer, index=True,nullable=True)
        expensiveprice = db.Column(db.Integer, index=True,nullable=True)
        volume = db.Column(db.Integer, index=True,nullable=True)
        dividendyield = db.Column(db.Integer, index=True,nullable=True)
        fiveyearavgdividendyield = db.Column(db.Integer, index=True,nullable=True)
        fiveyeardividend = db.Column(db.Integer, index=True,nullable=True)
        update = db.Column(db.Date, index=True,nullable=True)

        def to_dict4(self):
            return {
                'marketindexes': self.marketindexes,
                'symbol': self.symbol,
                'close': self.close,
                'cheapprice': self.cheapprice,
                'fairprice': self.fairprice,
                'expensiveprice': self.expensiveprice,
                'volume': self.volume,
                'dividendyield': self.dividendyield,
                'fiveyearavgdividendyield': self.fiveyearavgdividendyield,
                'fiveyeardividend': self.fiveyeardividend,
                'update': self.update
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

all_files = utils.read_all_stock_files('predict/individual_stocks_5yr')
@app.route('/predict.html')
def landing_function():
    title = "股價預測"
    # df = all_files['A']
    # df = pd.read_csv('GOOG_30_days.csv')
    # all_prediction_data, all_prediction_data, prediction_date, dates, all_data, all_data = perform_training('A', df, ['SVR_linear'])
    stock_files = ['AAPL', 'ADI', 'AMAT', 'AMZN', 'CVX', 'HD', 'JPM', 'MAR', 'TSLA', 'WMT']
    #stock_files = list(all_files.keys())
    return render_template('predict.html',show_results="false", title=title,
                           accuracy=[], test_score=float, prob_tomorrow=np.array([]),tommor=pd.DataFrame(),stock_files=stock_files, predict_data=pd.DataFrame(),pred_date_set=str, y_pred1=pd.DataFrame())

@app.route('/process', methods=['POST'])
def process():
    title = "股價預測"
    stock_file_name = request.form['stockfile']
    ml_algoritms = request.form.getlist('mlalgos')
    ml_algoritms = ml_algoritms[0]
    stockname = str(stock_file_name)
    df =pd.read_csv("traindata/"+stock_file_name+"_data_train.csv")
    df=df.drop(['symbol'],axis=1)
    if ml_algoritms=='KNN':
        accuracy, test_score, prob_tomorrow,tommor, predict_data, pred_date_set, original_pred_data,y_pred1=KNN(df,day_pred=60)
    elif ml_algoritms=='random_forests':
        accuracy, test_score, prob_tomorrow,tommor, predict_data, pred_date_set, original_pred_data,y_pred1=random_forests(df,day_pred=60) 
    stock_files = ['AAPL', 'ADI', 'AMAT', 'AMZN', 'CVX', 'HD', 'JPM', 'MAR', 'TSLA', 'WMT']
    #stock_files = list(all_files.keys())
    html_name = "/modelhtml/"+stock_file_name+"_"+str(ml_algoritms)+".html"
    return render_template('predict.html', all_test_evaluations=str(ml_algoritms), show_results="true",stock_files=stock_files,stock_file_name=stock_file_name,html_name=html_name,
                           accuracy=accuracy, test_score=test_score, prob_tomorrow=prob_tomorrow,tommor=tommor.updown, pred_date_set=str(pred_date_set.loc[60]), predict_data=predict_data, y_pred1=y_pred1)
#####模型視覺化區塊________
@app.route('/modelhtml/AAPL_KNN.html')
def html1():
	return render_template('/modelhtml/AAPL_KNN.html', title=title)
@app.route('/modelhtml/AAPL_random_forests.html')
def html2():
	return render_template('/modelhtml/AAPL_random_forests.html', title=title)
@app.route('/modelhtml/ADI_KNN.html')
def html3():
	return render_template('/modelhtml/ADI_KNN.html', title=title)
@app.route('/modelhtml/ADI_random_forests.html')
def html4():
	return render_template('/modelhtml/ADI_random_forests.html', title=title)
@app.route('/modelhtml/AMAT_KNN.html')
def html5():
	return render_template('/modelhtml/AMAT_KNN.html', title=title)
@app.route('/modelhtml/AMAT_random_forests.html')
def html6():
	return render_template('/modelhtml/AMAT_random_forests.html', title=title)
@app.route('/modelhtml/AMZN_KNN.html.html')
def html7():
	return render_template('/modelhtml/AMZN_KNN.html', title=title)
@app.route('/modelhtml/AMZN_random_forests.html')
def html8():
	return render_template('/modelhtml/AMZN_random_forests.html', title=title)
@app.route('/modelhtml/CVX_KNN.html')
def html9():
	return render_template('/modelhtml/CVX_KNN.html', title=title)
@app.route('/modelhtml/CVX_random_forests.html')
def html10():
	return render_template('/modelhtml/CVX_random_forests.html', title=title)
@app.route('/modelhtml/HD_KNN.html')
def html11():
	return render_template('/modelhtml/HD_KNN.html', title=title)
@app.route('/modelhtml/HD_random_forests.html')
def html12():
	return render_template('/modelhtml/HD_random_forests.html', title=title)
@app.route('/modelhtml/JPM_KNN.html')
def html13():
	return render_template('/modelhtml/JPM_KNN.html', title=title)
@app.route('/modelhtml/JPM_random_forests.html')
def html14():
	return render_template('/modelhtml/JPM_random_forests.html', title=title)
@app.route('/modelhtml/MAR_KNN.html')
def html15():
	return render_template('/modelhtml/MAR_KNN.html', title=title)
@app.route('/modelhtml/MAR_random_forests.html')
def html16():
	return render_template('/modelhtml/MAR_random_forests.html', title=title)
@app.route('/modelhtml/TSLA_KNN.html')
def html17():
	return render_template('/modelhtml/TSLA_KNN.html', title=title)
@app.route('/modelhtml/TSLA_random_forests.html')
def html18():
	return render_template('/modelhtml/TSLA_random_forests.html', title=title)
@app.route('/modelhtml/WMT_KNN.html')
def html19():
	return render_template('/modelhtml/WMT_KNN.html', title=title)
@app.route('/modelhtml/WMT_random_forests.html')
def html20():
	return render_template('/modelhtml/WMT_random_forests.html', title=title)

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