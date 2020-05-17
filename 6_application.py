from flask import Flask,send_from_directory,request
import logging
from pandas.io.json import dumps as jsonify
import os, random
from IPython.display import Javascript,HTML
from flask import Flask
from collections import ChainMap
from churnexplainer import ExplainedModel
# Imports needed for the churn explainer code.

# This reduces the the output to the console window
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

em = ExplainedModel(model_name='telco_linear',data_dir='/home/cdsw')

def explainid(N):
    customer_data = dataid(N)[0]
    customer_data.pop('id')
    customer_data.pop('Churn probability')
    data = em.cast_dct(customer_data)
    probability, explanation = em.explain_dct(data)
    return {'data': dict(data),
                    'probability': probability,
                    'explanation': explanation,
           'id':int(N)}  

def dataid(N):
    customer_id = em.data.index.dtype.type(N)
    customer_df = em.data.loc[[customer_id]].reset_index()
    return customer_df.to_dict(orient='records')

flask_app = Flask(__name__,static_url_path='')

@flask_app.route('/')
def home():
    return "<script> window.location.href = '/flask/table_view.html'</script>"

@flask_app.route('/flask/<path:path>')
def send_file(path):
    return send_from_directory('flask', path)

@flask_app.route('/sample_table')
def sample_table():
  #N = request.args.get('N', 10, int)
  sample_ids = random.sample(range(1,len(em.data)),10)
  sample_table = []
  for ids in sample_ids:
    sample_table.append(explainid(str(ids)))
  return jsonify(sample_table)

@flask_app.route("/categories")
def categories():
    return jsonify({feat: dict(enumerate(cats))
                   for feat, cats in em.categories.items()})

@flask_app.route("/stats")
def stats():
    return jsonify(em.stats)

HTML("<a href='https://{}.{}'>Open Table View</a>".format(os.environ['CDSW_ENGINE_ID'],os.environ['CDSW_DOMAIN']))

if __name__=="__main__":
    flask_app.run(host='127.0.0.1', port=int(os.environ['CDSW_APP_PORT']))