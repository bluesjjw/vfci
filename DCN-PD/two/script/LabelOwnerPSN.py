import torch
import sys

sys.path.append('..')
from models.Bottom_12features import Bottom_12features
from models.Bottom_13features import Bottom_13features
from models.TopPSN import TopPSN

import torch.optim as optim
import torch.nn.functional as F

from utils.args import args_parser

from data.data_partition import DataPartition
import torch.distributed as dist

import transmission.tenseal.tenseal_psndata_pb2 as tenseal_psndata_pb2
import transmission.tenseal.tenseal_psndata_pb2_grpc as tenseal_psndata_pb2_grpc

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


def forwardTop(network, forwardOut, treatment, optimizer):
    forwardOut = torch.tensor(forwardOut, requires_grad=True).cuda()

    rows = int(forwardOut.size(0) / 25)
    forwardOut = forwardOut.unflatten(0, (rows, 25))
    forwardOut.retain_grad()
    network.train()
    treatment_pred = network(forwardOut)
    treatment=treatment.cuda().squeeze()

    loss = F.cross_entropy(treatment_pred.cuda(), treatment.long().cuda())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    middle_grad = forwardOut.grad

    return middle_grad




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
    stub = tenseal_psndata_pb2_grpc.SafeTransmissionStub(channel)
    csv_path = "../data/2client/client{0}.csv".format(rank + 1)
    dL = DataPartition(csv_path, 0.8, 2)
    datadict = dL.getPSNTensor(csv_path, rank)
    ps_train_x = datadict["PSN_trainData"][1]

    dataset = torch.utils.data.TensorDataset(ps_train_x, datadict["PSN_trainData"][2])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    network = Bottom_12features().cuda()
    topmodel = TopPSN("train").cuda()

    model_save_path = ""
    checkpoint_path = ""
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    print(network)

    # checkpint=torch.load('')
    # network.load_state_dict(checkpint['net'])
    # optimizer.load_state_dict(checkpint['optimizer'])
    print(len(ps_train_x))
    for epo in tqdm(range(1, 51)):

        print("epoch: ", epo)
        rnd = 0
        wait_start = time.time()
        # for rnd in tqdm(range(len(ps_train_x))):
        for batch in data_loader:
            print(rnd)
            # dist.barrier()

            train_x, train_y = batch
            # client Bottom forward,get features
            out = forwardBottom(network, train_x)
            send_out = out.detach().cpu()
            send_out = send_out.flatten()
            # print(rank, out)
            plain_vector = ts.plain_tensor(send_out)
            encrypted_vector = ts.ckks_vector(ctx, plain_vector)
            serialize_msg = encrypted_vector.serialize()


            response = stub.MiddleForward(tenseal_psndata_pb2.encrypted(round=rnd, client_rank=rank, msg=serialize_msg))


            # receive ps_out from server and decrypt it
            enc_vector = ts.ckks_vector_from(ctx, response.msg)
            dec_vector = enc_vector.decrypt()


            middle_grad = forwardTop(topmodel, dec_vector,train_y,optimizer)

            middle_grad = middle_grad.cpu().flatten()
            # print(grad)
            plain_vector = ts.plain_tensor(middle_grad)
            encrypted_vector = ts.ckks_vector(ctx, plain_vector)
            serialize_msg = encrypted_vector.serialize()
            gresponse = stub.MiddleBackward(tenseal_psndata_pb2.gradenc(client_rank=rank, msg=serialize_msg))
            if gresponse.flag == 0:
                print("Client send loss grad occurs error")


            gradresponse = stub.BottomBackward(tenseal_psndata_pb2.sendgrad(client_rank=rank, round=rnd))
            enc_grad_vector = ts.ckks_vector_from(ctx, gradresponse.msg)
            dec_grad_vector = enc_grad_vector.decrypt()
            # print(rank, dec_grad_vector)

            # dist.barrier()

            dec_grad_vector = [round(i, 4) for i in dec_grad_vector]
            schegrad = torch.tensor(dec_grad_vector).cuda()

            rows = int(schegrad.size(0) / 12)
            schegrad = schegrad.unflatten(0, (rows, 12))

            backwardBottom(schegrad, optimizer, out)
            rnd += 1
        wait_time = time.time() - wait_start
        print("wait time", wait_time)
        # checkpoint = {'optimizer': optimizer.state_dict(), 'net': network.state_dict()}
        # torch.save(checkpoint, check_save_path.format(epo, rank))
        # if epo % 10 == 0 or epo == 51:
        #     torch.save(network.state_dict(), model_save_path.format(rank + 1, epo))


ctx_file = "../transmission/tenseal/ts_ckks_tiny.config"
context_bytes = open(ctx_file, "rb").read()
ctx = ts.context_from(context_bytes)
if __name__ == "__main__":
    init_processes(1, 2, run)
