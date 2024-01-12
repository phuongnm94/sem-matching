from flask import request, jsonify, Flask, render_template
import logging
import json, re

from flask.wrappers import Response
import sys
from pytorch_lightning import Trainer
from pre_processing.tree import word_matching_score, get_semantic_matching_score
from trainers.trainer_semparser import BertLabeling
from utils.utils import set_random_seed

set_random_seed(0)


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
PORT = 7009
HOST = "0.0.0.0"
sp_engine = None

trained_model = dict()
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, filename='./server_run.log',
                        filemode='a',)

def str_preprocess(str_in):
    return re.sub(r" {2,}", " ", str_in.strip()).strip()

@app.route('/sem-matching', methods=['GET', 'POST'])
def sem_matching():
    
    if request.method == "POST":
        json_dat = json.loads(request.data)
        question = json_dat.get('sentence_a')
        sentence_b = json_dat.get('sentence_b')
        
    elif request.method == "GET":
        question = request.args.get('sentence_a')
        sentence_b = request.args.get('sentence_b')

    out, w_matching_score, w_matching_detail, s_matching_score, s_matching_detail = __process(question, sentence_b)
     
    return jsonify({
                        "sentence_a": question, 
                        "sentence_b": sentence_b, 
                        "logic_a":out[0]['pred'], 
                        "logic_b":out[1]['pred'],
                        "sem_score": s_matching_score,
                        "word_matching_score":  w_matching_score,
                        "semantic_intersection": s_matching_detail ,
                        "word_matching_intersection":  w_matching_detail,
                    })

def __process(question, sentence_b):
    
    logger.info([question, sentence_b])
    out = __parse([question, sentence_b])
    
    w_matching_score, w_matching_detail = word_matching_score(question, sentence_b)
    s_matching_score, s_matching_detail = get_semantic_matching_score(out[0]['pred'], out[1]['pred'])
    return out, w_matching_score, w_matching_detail, s_matching_score, s_matching_detail
    
@app.route('/', methods=['GET', 'POST'])
def answer_form():
    dat = json.load(open('./data/top/mrc-ner.dev'))
    all_sents = [e['context'] for e in dat]
    question = all_sents[47]               
    sentence_b = all_sents[209]               
    
    if request.method == "GET":
        return render_template('greeting.html', 
                               question=question, 
                               sentence_b=sentence_b,
                               all_sents=all_sents)
    
    question = str_preprocess(request.form['question'])
    sentence_b = str_preprocess(request.form['sentence_b'])
    
    out, w_matching_score, w_matching_detail, s_matching_score, s_matching_detail = __process(question, sentence_b)
     
    return render_template('greeting.html', 
                           question=question, 
                           sentence_b=sentence_b, 
                           detail_ans1=out[0]['pred'], 
                           detail_ans2=out[1]['pred'],
                           sem_score=s_matching_score,
                           word_matching_score = w_matching_score,
                           semantic_intersection=json.dumps(s_matching_detail, indent=1),
                           word_matching_intersection=json.dumps(w_matching_detail, indent=1),
                           all_sents=all_sents
                           )

 
def jsonstr_return(data):
    json_response = json.dumps(data, ensure_ascii = False)
    #creating a Response object to set the content type and the encoding
    response = Response(json_response, content_type="application/json; charset=utf-8" )
    return response 

# @app.route('/answer', methods=['POST'])
# def translate():
#     inputs = request.get_json(force=True)
#     question = inputs.get("question")
#     logger.info(question)

#     out = __parse([question])

#     return jsonstr_return(out)

# @app.route('/answer-debug-detail', methods=['GET', 'POST'])
# def translate_debug_detail():
#     question = request.args.get("question") 
#     logger.info(question)

#     out = __parse([question])
    
#     return jsonstr_return(out)

# @app.route('/answer-debug', methods=['GET', 'POST'])
# def translate_debug():
#     question = request.args.get("question") 
#     logger.info(question)

#     out = __parse([question])
    
#     return jsonstr_return(out[0])

def __init_model(ckpt, hparams_file, gpus, max_length):
    global sp_engine
    
    trainer = Trainer(gpus=gpus, distributed_backend="dp")

    model = BertLabeling.load_from_checkpoint(
        checkpoint_path=ckpt,
        hparams_file=hparams_file,
        map_location=None,
        batch_size=1,
        max_length=max_length,
        workers=0
    )

    sp_engine = {'model' : model,
                 'trainer' : trainer
                 }
    return "-- Run model sucessful: "+ str(sp_engine)

def __parse(sentences=['what is the event near Ho Chi Minh city', 'how is the traffic in in Ha Noi , Vietnam']):
    global sp_engine
    model, trainer = sp_engine['model'], sp_engine['trainer']
    
    # trainer.test(model=model)
    model.write_test_file(sentences)
    trainer.test(model=model)
    return model.read_test_outputs()

"""

curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"question": "Who is the first president of US?"}' \
  http://150.65.183.93:7009/answer 


curl --header "Content-Type: application/json" \
--request POST \
--data '{"question": "what the boiling point of water?"}' \
http://150.65.183.93:7009/answer 

"""

if __name__ == "__main__":
    
    print(sys.argv)
    CHECKPOINTS = sys.argv[1]
    HPARAMS = sys.argv[2]
    try:
        GPUS = [int(gpu_item) for gpu_item in sys.argv[3].strip().split(",")]
    except:
        GPUS = [0]

    try:
        MAXLEN = int(sys.argv[4])
    except:
        MAXLEN = 512


    logger.info(__init_model(ckpt=CHECKPOINTS, hparams_file=HPARAMS, gpus=GPUS, max_length=MAXLEN))
    
    __parse()
    # __parse()
    app.run(host=HOST, port=PORT)