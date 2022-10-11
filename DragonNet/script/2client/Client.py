import torch
import sys

sys.path.append('..')
from models.client_num2.Bottom_13features import Bottom_13features
from models.client_num2.Bottom_12features import Bottom_12features
import torch.optim as optim

from utils.args import args_parser

from data.client_num2.data_partition import DataPartition
import torch.distributed as dist
import transmission.tenseal.tenseal_DragonNetData_pb2 as tenseal_DragonNetData_pb2
import transmission.tenseal.tenseal_DragonNetData_pb2_grpc as tenseal_DragonNetData_pb2_grpc
from models.utils import *

args = args_parser()
import tenseal as ts

import grpc
from tqdm import tqdm
import time


def forwardBottom(network, train_x):
    # print(".. Training started ..")

    network.train()

    train_x = train_x.cuda()
    features = network(train_x).cuda()

    return features


def backwardBottom(grad, optimizer, ps_out):
    optimizer.zero_grad()
    ps_out.backward(grad.cuda())
    optimizer.step()



def init_processes(rank, size, fn):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend='gloo',
                            init_method="tcp://127.0.0.1:12345",
                            rank=rank,
                            world_size=size)
    fn(rank)


def run(rank):
    max_msg_size = 1000000000
    options = [('grpc.max_send_message_length', max_msg_size),
               ('grpc.max_receive_message_length', max_msg_size)]
    channel = grpc.insecure_channel('127.0.0.1:50051', options=options)
    stub = tenseal_DragonNetData_pb2_grpc.SafeTransmissionStub(channel)
    csv_path = "../../data/client_num2/client{0}.csv".format(rank + 1)
    dL = DataPartition(csv_path, 0.8, 2)
    datadict = dL.getDragonNetTensor(csv_path, rank)
    idx_train = datadict["DragonNet_trainData"][0]
    dragonNet_train_x = datadict["DragonNet_trainData"][1]

    dataset = torch.utils.data.TensorDataset(idx_train, dragonNet_train_x)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    network = Bottom_13features().cuda()
    model_save_path = ""
    check_save_path = ""

    optimizer_adam = optim.Adam(network.parameters(), lr=1e-3,weight_decay=0.01)
    optimizer = optimizer_adam
    network.train()

    # checkpint=torch.load('../checkpoints/schepsn/2client_all_features_epoch_50_distPSN{0}_checkpoint.pth'.format(rank))
    # network.load_state_dict(checkpint['net'])
    # optimizer.load_state_dict(checkpint['optimizer'])
    print(len(dragonNet_train_x))
    for epo in tqdm(range(1, 401)):
        dist.barrier()
        if epo == 201:
            optimizer_sgd = optim.SGD(network.parameters(), lr=1e-5, momentum=0.9, nesterov=True, weight_decay=0.01)
            optimizer = optimizer_sgd
        print("epoch: ", epo)
        rnd = 0
        wait_start = time.time()
        # for rnd in tqdm(range(len(ps_train_x))):
        for batch in data_loader:
            print(rnd)
            dist.barrier()
            idx, train_x = batch

            # client bottom forward,get features
            out = forwardBottom(network, train_x)
            send_out = out.detach().cpu()
            send_out = send_out.flatten()
            # print(rank, out)
            plain_vector = ts.plain_tensor(send_out)
            encrypted_vector = ts.ckks_vector(ctx, plain_vector)
            serialize_msg = encrypted_vector.serialize()

            dist.barrier()
            response = stub.MiddleForward(
                tenseal_DragonNetData_pb2.encrypted(round=rnd, client_rank=rank, msg=serialize_msg))
            dist.barrier()

            # receive ps_out from server and decrypt it
            enc_vector = ts.ckks_vector_from(ctx, response.msg)
            dec_vector = enc_vector.decrypt()


            gradresponse = stub.BottomBackward(tenseal_DragonNetData_pb2.sendgrad(client_rank=rank, round=rnd))
            enc_grad_vector = ts.ckks_vector_from(ctx, gradresponse.msg)
            dec_grad_vector = enc_grad_vector.decrypt()
            # print(rank, dec_grad_vector)

            # dec_grad_vector = [round(i, 4) for i in dec_grad_vector]
            schegrad = torch.tensor(dec_grad_vector).cuda()

            rows = int(schegrad.size(0) / 104)
            schegrad = schegrad.unflatten(0, (rows, 104))

            backwardBottom(schegrad, optimizer, out)
            rnd += 1

        wait_time = time.time() - wait_start
        print("wait time: ",wait_time)
        checkpoint = {'optimizer': optimizer.state_dict(), 'net': network.state_dict()}
        torch.save(checkpoint, check_save_path.format(epo, rank))

        if epo % 10 == 0 or epo == 201:
            torch.save(network.state_dict(), model_save_path.format(rank + 1, epo))


ctx_file = "../../transmission/tenseal/ts_ckks_tiny.config"
context_bytes = open(ctx_file, "rb").read()
ctx = ts.context_from(context_bytes)
if __name__ == "__main__":
    init_processes(0, 2, run)

