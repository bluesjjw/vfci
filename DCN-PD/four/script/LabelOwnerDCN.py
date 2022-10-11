import torch
import sys

sys.path.append('..')
from models.Bottom_12features import Bottom_12features
from models.TopDCN import TopDCN
from models.TopPSN import TopPSN
import torch.optim as optim
from transmission.Utils import forwardTop
import torch.nn as nn
from utils.args import args_parser

from data.data_partition import DataPartition
import torch.distributed as dist

import os
import transmission.tenseal.tenseal_dcndata_pb2 as tenseal_psndata_pb2
import transmission.tenseal.tenseal_dcndata_pb2_grpc as tenseal_dcndata_pb2_grpc

args = args_parser()
import tenseal as ts
import time
import grpc


def forwardBottom(bottom, train_x):
    # print(".. Training started ..")

    train_x = train_x.cuda()
    features = bottom(train_x).cuda()

    return features


def getPscore(topmodel, features):
    features = torch.tensor(features, requires_grad=True).cuda()
    pscore = topmodel(features)
    prob = pscore[1]
    return prob


def init_processes(rank, size, fn):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='gloo',
                            rank=rank,
                            world_size=size)
    csv_path = '../data/2client/client{0}.csv'.format(rank + 1)
    dp = DataPartition(csv_path, 0.8, 2)

    dataset = dp.getDCNTensorWithLabel(csv_path, "train")
    print("client{0} load data successfully".format(rank + 1))

    fn(rank, dataset)


def run(rank, dataset):
    # 注册grpc
    max_msg_size = 1000000000
    options = [('grpc.max_send_message_length', max_msg_size),
               ('grpc.max_receive_message_length', max_msg_size)]
    channel = grpc.insecure_channel('localhost:50051', options=options)
    stub = tenseal_dcndata_pb2_grpc.SafeTransmissionStub(channel)
    bottom = Bottom_12features().cuda()
    topDCN = TopDCN(training_flag=True).cuda()
    optimizer = optim.Adam(topDCN.parameters(), lr=0.001)
    topPSN = TopPSN("eval").cuda()
    # initialize parameters and load bottom

    bottom_path = ''
    top_path = ''
    bottom_dict = torch.load(bottom_path)
    bottom.load_state_dict(bottom_dict)
    top_dict = torch.load(top_path)
    topPSN.load_state_dict(top_dict)

    # load data
    treated_data = dataset["treat_trainData"]
    control_data = dataset["control_trainData"]
    treated_idx = treated_data[0]
    control_idx = control_data[0]
    treated_x = treated_data[1]
    control_x = control_data[1]

    for epo in range(1, 61):
        print("epoch ", epo)
        dist.barrier()
        wait_start = time.time()
        if epo % 2 == 0:
            # train treat data
            for rnd in range(len(treated_x)):
                print("id", treated_idx[rnd])

                # client Bottom forward,get features
                out = forwardBottom(bottom, treated_x[rnd])
                send_out = out.detach().cpu()
                # print(rank, out)

                # encrypt the features and send to server
                plain_vector = ts.plain_tensor(send_out)
                encrypted_vector = ts.ckks_vector(ctx, plain_vector)
                serialize_msg = encrypted_vector.serialize()
                response = stub.MiddleForward(
                    tenseal_psndata_pb2.encrypted(round=rnd, client_rank=rank, epoch=epo, msg=serialize_msg))

                # send propensity score to server
                enc_vector = ts.ckks_vector_from(ctx, response.msg)
                dec_vector = enc_vector.decrypt()
                pscore = getPscore(topPSN, dec_vector)

                plain_vector = ts.plain_tensor(pscore)
                encrypted_vector = ts.ckks_vector(ctx, plain_vector)
                serialize_msg = encrypted_vector.serialize()
                pscore_response = stub.PscoreTransmit(
                    tenseal_psndata_pb2.encrypted(round=rnd, client_rank=rank, epoch=epo, msg=serialize_msg))

                enc_vector = ts.ckks_vector_from(ctx, pscore_response.msg)
                dec_vector = enc_vector.decrypt()
                # predict_ITE = torch.tensor(dec_vector, requires_grad=True)

                # features ,calculate True ITE
                outcome_y = treated_data[3]
                y_f, y_cf = outcome_y[rnd].chunk(2, 0)
                true_ITE = y_f - y_cf

                # calculate grad and encrypt it to send to server
                grad = forwardTop(epo, dec_vector, pscore, topDCN, true_ITE, optimizer).cpu()
                # print(grad)
                plain_vector = ts.plain_tensor(grad)
                encrypted_vector = ts.ckks_vector(ctx, plain_vector)
                serialize_msg = encrypted_vector.serialize()
                gresponse = stub.MiddleBackward(tenseal_psndata_pb2.gradenc(client_rank=rank, msg=serialize_msg))

                if gresponse.flag == 0:
                    print("Client send loss grad occurs error")

        if epo % 2 == 1:
            # train control data
            for rnd in range(len(control_x)):

                print(control_idx[rnd])
                dist.barrier()
                # client Bottom forward,get features
                out = forwardBottom(bottom, control_x[rnd])
                send_out = out.detach().cpu()
                # print(rank, out)

                # encrypt the features and send to server
                plain_vector = ts.plain_tensor(send_out)
                encrypted_vector = ts.ckks_vector(ctx, plain_vector)
                serialize_msg = encrypted_vector.serialize()
                response = stub.MiddleForward(
                    tenseal_psndata_pb2.encrypted(round=rnd, client_rank=rank, epoch=epo, msg=serialize_msg))

                # send propensity score to server
                enc_vector = ts.ckks_vector_from(ctx, response.msg)
                dec_vector = enc_vector.decrypt()
                pscore = getPscore(topPSN, dec_vector)

                plain_vector = ts.plain_tensor(pscore)
                encrypted_vector = ts.ckks_vector(ctx, plain_vector)
                serialize_msg = encrypted_vector.serialize()
                pscore_response = stub.PscoreTransmit(
                    tenseal_psndata_pb2.encrypted(round=rnd, client_rank=rank, epoch=epo, msg=serialize_msg))

                enc_vector = ts.ckks_vector_from(ctx, pscore_response.msg)
                dec_vector = enc_vector.decrypt()

                outcome_y = control_data[3]
                y_f, y_cf = outcome_y[rnd].chunk(2, 0)
                true_ITE = y_cf - y_f
                # calculate grad and encrypt it to send to server
                grad = forwardTop(epo, dec_vector, pscore, topDCN, true_ITE, optimizer).cpu()
                # print(grad)
                plain_vector = ts.plain_tensor(grad)
                encrypted_vector = ts.ckks_vector(ctx, plain_vector)
                serialize_msg = encrypted_vector.serialize()
                gresponse = stub.MiddleBackward(tenseal_psndata_pb2.gradenc(client_rank=rank, msg=serialize_msg))

                if gresponse.flag == 0:
                    print("Client send loss grad occurs error")
        wait_time = time.time() - wait_start
        print("wait time: ", wait_time)


ctx_file = "../transmission/tenseal/ts_ckks_tiny.config"
context_bytes = open(ctx_file, "rb").read()
ctx = ts.context_from(context_bytes)
if __name__ == "__main__":
    init_processes(1, 2, run)
