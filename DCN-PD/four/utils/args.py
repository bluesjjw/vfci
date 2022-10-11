#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n',
                        default="default",
                        type=str,
                        help='experiment name, used for saving results')
    parser.add_argument('--backend',
                        default="gloo",
                        type=str,
                        help='background name')
    parser.add_argument('--model',
                        default="DCN",
                        type=str,
                        help='neural network model')
    parser.add_argument('--alpha',
                        default=0.2,
                        type=float,
                        help='control the non-iidness of dataset')
    parser.add_argument('--gmf',
                        default=0,
                        type=float,
                        help='global (server) momentum factor')
    parser.add_argument('--lr',
                        default=0.001,
                        type=float,
                        help='DCN client learning rate')
    parser.add_argument('--plr',
                        default=0.001,
                        type=float,
                        help='PSNet client learning rate')
    parser.add_argument('--momentum',
                        default=0.0,
                        type=float,
                        help='local (client) momentum factor')
    parser.add_argument('--pbs',
                        default=8,
                        type=int,
                        help='batch size on each worker/client')
    parser.add_argument('--rounds',
                        default=10,
                        type=int,
                        help='total coommunication rounds')
    parser.add_argument('--localE',
                        default=10,
                        type=int,
                        help='number of DCN local epochs')
    parser.add_argument('--PSnetEpoch',
                        default=50,
                        type=int,
                        help='number of PSNet local epochs')
    parser.add_argument('--print_freq',
                        default=100,
                        type=int,
                        help='print info frequency')
    parser.add_argument('--size',
                        default=2,
                        type=int,
                        help='number of local workers')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='the rank of worker')
    parser.add_argument('--idx',
                        default=0,
                        type=int,
                        help='the id of dataset')
    parser.add_argument('--seed',
                        default=1,
                        type=int,
                        help='random seed')
    parser.add_argument('--save', '-s',
                        action='store_true',
                        help='whether save the training results')
    parser.add_argument('--p', '-p',
                        action='store_true',
                        help='whether the dataset is partitioned or not')
    parser.add_argument('--NIID',
                        action='store_true',
                        help='whether the dataset is non-iid or not')
    parser.add_argument('--pattern',
                        type=str,
                        help='pattern of local steps')
    parser.add_argument('--optimizer',
                        default='local',
                        type=str,
                        help='optimizer name')
    parser.add_argument('--initmethod',
                        default='tcp://127.0.0.1:29500',
                        type=str,
                        help='init method')
    parser.add_argument('--mu',
                        default=0,
                        type=float,
                        help='mu parameter in fedprox')
    parser.add_argument('--savepath',
                        default='./save/DCNModel',
                        type=str,
                        help='directory to save exp results')
    parser.add_argument('--savePSNet',
                        default='../save/PSNet',
                        type=str,
                        help='directory to save exp results')
    parser.add_argument('--datapath',
                        default='./data/Dataset',
                        type=str,
                        help='directory to load data')
    args = parser.parse_args()
    return args
