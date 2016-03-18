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
logging.basicConfig(level=logging.INFO)

from flask.ext.cors import cross_origin
from controler import Controler, Logger

app = Flask(__name__)
# app.config['CORS_HEADERS'] = 'Content-Type'


def request2json(data):
    return json.loads(data.decode('utf8'))


@app.route('/listmodel', methods=['GET'])
def list_model():
    print controler.list_model()
    return json.dumps(controler.list_model())

@app.route('/predict', methods=['GET'])
@cross_origin()
def predict():
    model = request.args['model']
    text = request.args['text']
    print text
    res = controler.predict(model, text)
    return json.dumps(res)

@app.route('/log', methods=['GET'])
@cross_origin()
def log():
    data = json.loads(request.args['data'])
    return json.dumps(logger.log(data))

# @app.route('/log', methods=['POST'])
# @cross_origin()
# def log():
#     data = request2json(request.data)
#     return json.dumps(logger.log(data))

# @app.route('/predict', methods=['POST'])
# @cross_origin()
# def predict():
#     print '='*40
#     print request
#     print '='*40
#     print request.data
#     print '='*40
#     data = request2json(request.data)
#     print data
#     res = controler.predict(data['model'], data['text'])
#     return json.dumps(res)


# @app.route('/log', methods=['POST'])
# @cross_origin()
# def log():
#     data = request2json(request.data)
#     return json.dumps(logger.log(data))


def parse_arg(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-a', '--address', default='0.0.0.0',
                        help='accesible address')
    parser.add_argument('-p', '--port', default=5125, type=int, help='port')
    parser.add_argument(
        '-d', '--debug', action='store_true', help='debug mode')

    return parser.parse_args(argv[1:])

if __name__ == "__main__":
    args = parse_arg(sys.argv)
    controler = Controler(os.path.abspath('../model'))
    logger = Logger()

    if args.debug:
        args.port += 1
        app.run(args.address, args.port, debug='true',
                threaded=True, use_reloader=False)
    else:
        print "* Running on http://{}:{}/".format(args.address, args.port)
        WSGIServer(app, multithreaded=True, bindAddress=(
            args.address, args.port)).run()
