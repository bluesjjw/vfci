import torch
import sys

sys.path.append('..')

from models.Bottom_13features import Bottom_13features

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


def forwardBottom(network, train_x):
    # print(".. Training started ..")

    train_x = train_x.cuda()
    features = network(train_x).cuda()

    return features


def init_processes(rank, size, fn):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='gloo',
                            rank=rank,
                            world_size=size)
    csv_path = '../data/2client/client{0}.csv'.format(rank + 1)
    dp = DataPartition(csv_path, 0.8, 2)

    index = dp.getIndex('../data/2client/client2.csv')
    dataset = dp.getDCNTensorWithoutLabel(csv_path, "train", index)
    print("client{0} load data successfully".format(rank + 1))

    fn(rank, dataset)



def run(rank, dataset):
    # 注册grpc
    max_msg_size = 1000000000
    options = [('grpc.max_send_message_length', max_msg_size),
               ('grpc.max_receive_message_length', max_msg_size)]
    channel = grpc.insecure_channel('localhost:50051', options=options)
    stub = tenseal_dcndata_pb2_grpc.SafeTransmissionStub(channel)
    network = Bottom_13features().cuda()
    # initialize parameters and load network
    model_path = ''.format(rank + 1)
    state_dict = torch.load(model_path)
    network.load_state_dict(state_dict)
    print("rank: ", rank, network)

    # load data
    treated_data = dataset["treat_trainData"]
    control_data = dataset["control_trainData"]
    treated_idx = treated_data[0]
    control_idx = control_data[0]
    treated_x = treated_data[1]
    control_x = control_data[1]

    for epo in range(1, 61):
        print("epoch ", epo)
        wait_start =time.time()
        if epo % 2 == 0:
            # train treat data
            for rnd in range(len(treated_x)):
                print("id", treated_idx[rnd])
                # client Bottom forward,get features
                out = forwardBottom(network, treated_x[rnd])
                send_out = out.detach().cpu()

                # encrypt the features and send to server
                plain_vector = ts.plain_tensor(send_out)
                encrypted_vector = ts.ckks_vector(ctx, plain_vector)
                serialize_msg = encrypted_vector.serialize()
                response = stub.MiddleForward(
                    tenseal_psndata_pb2.encrypted(round=rnd, client_rank=rank, epoch=epo, msg=serialize_msg))

        if epo % 2 == 1:
            # train control data
            for rnd in range(len(control_x)):

                print(control_idx[rnd])
                dist.barrier()
                # client Bottom forward,get features
                out = forwardBottom(network, control_x[rnd])
                send_out = out.detach().cpu()
                # print(rank, out)

                # encrypt the features and send to server
                plain_vector = ts.plain_tensor(send_out)
                encrypted_vector = ts.ckks_vector(ctx, plain_vector)
                serialize_msg = encrypted_vector.serialize()
                response = stub.MiddleForward(
                    tenseal_psndata_pb2.encrypted(round=rnd, client_rank=rank, epoch=epo,
                                                  msg=serialize_msg))

        wait_time = time.time()-wait_start
        print("wait time: ",wait_time)

ctx_file = "../ransmission/tenseal/ts_ckks_tiny.config"
context_bytes = open(ctx_file, "rb").read()
ctx = ts.context_from(context_bytes)

if __name__ == "__main__":
    init_processes(0, 4, run)
