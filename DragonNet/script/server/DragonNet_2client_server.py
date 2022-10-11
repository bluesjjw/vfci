from concurrent import futures
import sys

sys.path.append("..")
sys.path.append("../..")
import torch
import torch.optim as optim
import time
import transmission.tenseal.tenseal_DragonNetData_pb2 as tenseal_DragonNetData_pb2
import transmission.tenseal.tenseal_DragonNetData_pb2_grpc as tenseal_DragonNetData_pb2_grpc

import tenseal as ts
import grpc
from transmission.Utils import forwardMiddle, backwardMiddle, splitGrad
from models.client_num2.MiddleModel import MiddleModel
# from tkinter import _flatten



class DataTransServer(tenseal_DragonNetData_pb2_grpc.SafeTransmissionServicer):

    def __init__(self, address, num_clients, ctx_file):
        self.address = address
        self.num_clients = num_clients
        self.ctx_file = ctx_file
        context_bytes = open(self.ctx_file, "rb").read()
        self.ctx = ts.context_from(context_bytes)
        self.sleep_time = 0.1
        self.DragonNet = MiddleModel().cuda()
        # self.DragonNet.load_state_dict(
        #     torch.load('../../save/2client_all_features/2client_all_features_epoch_200_DragonNet.pth'))
        self.optimizer_adam = optim.Adam(self.DragonNet.parameters(), lr=1e-3, weight_decay=0.01)
        self.optimizer = self.optimizer_adam
        self.features = None
        self.middle_out = None
        self.grad_list = []

        # cache and counter for sum operation
        self.n_sum_round = 0
        self.decdict = {}
        self.vectors_list = []
        self.sum_data = []
        self.n_sum_request = 0
        self.n_sum_response = 0
        self.forward_completed = False
        self.epoch = 1
        self.model_save_path = ""
        self.check_save_path = ""

        #
        # self.checkpoint=torch.load('../../checkpoints/compsn/2client_all_features_epoch_50_mergePSN_checkpoint.pth')
        # self.DragonNet.load_state_dict(self.checkpoint['net'])
        # self.optimizer.load_state_dict(self.checkpoint["optimizer"])

    def reset_sum(self):
        self.vectors_list.clear()
        self.sum_data = []
        self.n_sum_request = 0
        self.n_sum_response = 0
        self.decdict.clear()
        self.forward_completed = False

    def reset_grad(self):
        self.grad_list.clear()
        self.n_sum_response = 0
        if self.epoch == 201:
            self.optimizer = optim.SGD(self.DragonNet.parameters(), lr=1e-5, momentum=0.9, nesterov=True,
                                       weight_decay=0.01)

    def sortResdict(self):
        sortlist = sorted(self.decdict)
        for i in sortlist:
            self.vectors_list.append(self.decdict[i])

    def getFeatures(self):
        features_list = self.vectors_list[0]
        flatten_features = torch.tensor(features_list, requires_grad=True)
        rows = int(flatten_features.size(0) / 104)
        features = flatten_features.unflatten(0, (rows, 104))
        # print(features)
        for client in range(1, self.num_clients):
            features_list = self.vectors_list[client]
            flatten_features = torch.tensor(features_list)
            rows = int(flatten_features.size(0) / 96)
            unflatten_features = flatten_features.unflatten(0, (rows, 96))
            features = torch.cat((features, unflatten_features), dim=-1)
        # print(features)

        return features

    def MiddleForward(self, request, context):
        # decrypt the vector form client
        enc = request.msg
        enc_vector = ts.ckks_vector_from(self.ctx, enc)
        dec_vector = enc_vector.decrypt()
        # dec_vector = [round(i, 4) for i in dec_vector]
        client_rank = request.client_rank
        rnd = request.round
        # add received data to cache

        # self.decdict[request.client_rank] = torch.tensor(dec_vector)
        self.decdict[request.client_rank] = dec_vector
        self.n_sum_request += 1

        # wait until receiving of all clients' requests
        wait_start = time.time()
        while self.n_sum_request % self.num_clients != 0:
            time.sleep(self.sleep_time)
        wait_time = time.time() - wait_start

        # Splicing all params,get total params for model
        if client_rank == self.num_clients - 1:
            self.sortResdict()
            self.features = self.getFeatures()
            self.middle_out = forwardMiddle(self.DragonNet, self.features)
            # print(self.middle_out)
            self.forward_completed = True

        while not self.forward_completed:
            time.sleep(self.sleep_time)

        plain_vector = ts.plain_tensor(self.middle_out.detach().cpu().flatten())
        encrypted_vector = ts.ckks_vector(self.ctx, plain_vector)
        serialize_msg = encrypted_vector.serialize()

        response = tenseal_DragonNetData_pb2.encrypted(round=1, client_rank=1, msg=serialize_msg)
        self.n_sum_response = self.n_sum_response + 1
        while self.n_sum_response % self.num_clients != 0:
            time.sleep(self.sleep_time)

        if client_rank == self.num_clients - 1:
            self.reset_sum()
        if rnd == 37 and client_rank == 1:
            checkpoint = {'optimizer': self.optimizer.state_dict(), 'net': self.DragonNet.state_dict()}
            torch.save(checkpoint, self.check_save_path.format(self.epoch))
            print("epoch:", self.epoch)
            if self.epoch % 10 == 0 or self.epoch == 51:
                torch.save(self.DragonNet.state_dict(), self.model_save_path.format(self.epoch))
            self.epoch += 1
        # wait until cache for sum is reset
        self.n_sum_round = self.n_sum_round + 1
        while self.n_sum_round % self.num_clients != 0:
            time.sleep(self.sleep_time)

        return response

    def MiddleBackward(self, request, context):
        # decrypt the vector form cl
        enc = request.msg
        enc_vector = ts.ckks_vector_from(self.ctx, enc)
        dec_vector = enc_vector.decrypt()
        # dec_vector = [round(i, 4) for i in dec_vector]
        client_rank = request.client_rank
        ingrad = torch.tensor(dec_vector).cuda()
        rows = int(ingrad.size(0) / 200)
        ingrad = ingrad.unflatten(0, (rows, 200))

        outgrad = backwardMiddle(ingrad, self.optimizer, self.middle_out, self.features)
        num_list = [104, 96]
        self.grad_list = splitGrad(outgrad.cpu(), num_list)
        # print(self.grad_list)
        flag = 0
        if self.grad_list != []:
            flag = 1

        response = tenseal_DragonNetData_pb2.sendgradflag(flag=flag)
        return response

    def BottomBackward(self, request, context):
        self.n_sum_request += 1

        # wait until receiving of all clients' requests

        while self.n_sum_request % self.num_clients != 0:
            time.sleep(self.sleep_time)
        client_rank = request.client_rank
        rnd = request.round
        plain_vector = ts.plain_tensor(self.grad_list[client_rank].flatten())
        encrypted_vector = ts.ckks_vector(self.ctx, plain_vector)
        serialize_msg = encrypted_vector.serialize()

        response = tenseal_DragonNetData_pb2.encrypted(round=rnd, client_rank=client_rank, msg=serialize_msg)
        self.n_sum_response = self.n_sum_response + 1
        while self.n_sum_response % self.num_clients != 0:
            time.sleep(self.sleep_time)

        if client_rank == self.num_clients - 1:
            self.reset_grad()
            self.reset_sum()

            # wait until cache for sum is reset
        self.n_sum_round = self.n_sum_round + 1
        while self.n_sum_round % self.num_clients != 0:
            time.sleep(self.sleep_time)
        return response


def serve():
    max_msg_size = 1000000000
    ctx_file = "../../transmission/tenseal/ts_ckks_tiny.config"
    options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5), options=options)
    tenseal_DragonNetData_pb2_grpc.add_SafeTransmissionServicer_to_server(DataTransServer(1, 2, ctx_file), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("grpc server start...")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
