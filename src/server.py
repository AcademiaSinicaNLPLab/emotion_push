#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from flup.server.fcgi import WSGIServer
import argparse
from flask import Flask
from flask import request
import json
import os
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)

from flask_cors import CORS
from controler import Controler, Logger

app = Flask(__name__)
CORS(app)

def request2json(data):
    return json.loads(data.decode('utf8'))


@app.route('/listmodel', methods=['GET'])
def list_model():
    return json.dumps(controler.list_model())

@app.route('/predict', methods=['POST'])
def predict():
    data = request2json(request.data)
    res = controler.predict(data['model'], data['text'])
    print data['text']
    print emotions[np.argsort(np.array(res['res']))[::-1]]
    return json.dumps(res)

@app.route('/log', methods=['POST'])
def log():
    try:
        data = request2json(request.data)
        return json.dumps(logger.log(data))
    except Exception as e:
        print "log failure"
        return "Failure"

def parse_arg(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-a', '--address', default='0.0.0.0', help='accesible address')
    parser.add_argument('-p', '--port', default=5125, type=int, help='port')
    parser.add_argument('-d', '--debug', action='store_true', help='debug mode')

    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_arg(sys.argv)
    controler = Controler(os.path.abspath('../model'))
    emotions = np.array(controler.list_model()['LJ40K_svm'])
    logger = Logger()

    if args.debug:
        args.port += 1
        app.run(args.address, args.port, debug='true',
                threaded=True, use_reloader=False)
    else:
        print "* Running on http://{}:{}/".format(args.address, args.port)
        WSGIServer(app, multithreaded=True, bindAddress=(args.address, args.port)).run()
