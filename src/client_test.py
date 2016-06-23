#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import json
import numpy as np
import argparse
import sys


preditc_text = [
    'Hello, how are you?',
    'I am ok.',
    'Please check the calander to see if you have time',
    'I am looking forward to it.',
    'I hope that you will come with me.',
    'He is expected to win the champion',
    'I can not stand with you anymore.',
    'I hate his bad behabior.',
    'He is such a jerk',
    'I fear that it will rain tomorrow',
    'He is going to kill us',
    'I will not able to pass the exam',
    'I am so glad that you come to my house',
    'We should celebrate for the team to win the game',
    'I am so excited about the coming concert.',
    'Cry, my dog just die',
    'No!!! I failed my exam.',
    'I am so lonely!!!',
    'I am so sleepy and I just want to lie on bed.',
    'I am drained out by the paper.',
    'I just can not do it anymore',
    'Vous êtes un gars mal',
]


# model = 'MR_svm'
# model = 'MR_cnn'
model = 'LJ40K_svm'
predict_d = [{'model':model, 'text': t} for t in preditc_text]

def parse_arg(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--debug', action='store_true', help='debugging url')
    parser.add_argument('-g', '--get', action='store_true', help='use get')
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_arg(sys.argv)
    if args.debug:
        url = 'http://penguin.iis.sinica.edu.tw:5126/'
    else:
        url = 'https://doraemon.iis.sinica.edu.tw/mimansa/'

    log_d = json.dumps({'uid':'test','response_time':100})

    emotions = np.array(json.loads(requests.get(url+'listmodel').text)[model])
    for p_d in predict_d:
        print p_d['text']
        if args.get:
            get_url = url+'predict'+'?model={}&text="{}"'.format(p_d['model'], p_d['text'])
            res = requests.get(get_url).text
        else:
            res = requests.post(url+'predict', data=json.dumps(p_d)).text

        pred = json.loads(res)['res']
        print emotions[np.argsort(np.array(pred))[::-1]], res

    # print requests.post(url+'log', data=log_d).text
