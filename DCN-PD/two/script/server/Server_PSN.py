from concurrent import futures
import sys

sys.path.append("..")
sys.path.append("../..")
import torch
import torch.optim as optim
import time
import transmission.tenseal.tenseal_psndata_pb2 as tenseal_psndata_pb2
import transmission.tenseal.tenseal_psndata_pb2_grpc as tenseal_psndata_pb2_grpc
import tenseal as ts
import grpc
from transmission.Utils import forwardComPSN, backwardComPSN, splitGrad
from models.MiddlePSN import MiddlePSN
from tkinter import _flatten


class DataTransServer(tenseal_psndata_pb2_grpc.SafeTransmissionServicer):

    def __init__(self, address, num_clients, ctx_file):
        self.address = address
        self.num_clients = num_clients
        self.ctx_file = ctx_file
        context_bytes = open(self.ctx_file, "rb").read()
        self.ctx = ts.context_from(context_bytes)
        self.sleep_time = 0.1
        self.PSNmodel = MiddlePSN().cuda()
        self.optimizer = optim.Adam(self.PSNmodel.parameters(), lr=0.001)
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
        # self.checkpoint=torch.load('')
        # self.PSNmodel.load_state_dict(self.checkpoint['net'])
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

    def sortResdict(self):
        sortlist = sorted(self.decdict)
        for i in sortlist:
            self.vectors_list.append(self.decdict[i])

    def getFeatures(self):
        features_list = self.vectors_list[0]
        flatten_features = torch.tensor(features_list,requires_grad=True)
        rows = int(flatten_features.size(0) / 13)
        features = flatten_features.unflatten(0, (rows, 13))
        # print(features)
        for client in range(1, self.num_clients):
            features_list = self.vectors_list[client]
            flatten_features = torch.tensor(features_list)
            rows = int(flatten_features.size(0) / 12)
            unflatten_features = flatten_features.unflatten(0, (rows, 12))
            features = torch.cat((features, unflatten_features), dim=-1)
        # print(features)

        return features

    def MiddleForward(self, request, context):
        # decrypt the vector form client
        enc = request.msg
        enc_vector = ts.ckks_vector_from(self.ctx, enc)
        dec_vector = enc_vector.decrypt()
        dec_vector = [round(i, 4) for i in dec_vector]
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

            # features = list(_flatten(self.vectors_list))
            # # print(len(features))
            # # print(features)
            #
            # self.features = torch.tensor(features, requires_grad=True).cuda()
            self.features = self.getFeatures()
            self.middle_out = forwardComPSN(self.PSNmodel, self.features)
            # print(self.middle_out)
            self.forward_completed = True

        while not self.forward_completed:
            time.sleep(self.sleep_time)

        plain_vector = ts.plain_tensor(self.middle_out.detach().cpu().flatten())
        encrypted_vector = ts.ckks_vector(self.ctx, plain_vector)
        serialize_msg = encrypted_vector.serialize()

        response = tenseal_psndata_pb2.encrypted(round=1, client_rank=1, msg=serialize_msg)
        self.n_sum_response = self.n_sum_response + 1
        while self.n_sum_response % self.num_clients != 0:
            time.sleep(self.sleep_time)

        if client_rank == self.num_clients - 1:
            self.reset_sum()
        # if rnd == 74 and client_rank == 1:
        #     checkpoint = {'optimizer': self.optimizer.state_dict(), 'net': self.PSNmodel.state_dict()}
        #     torch.save(checkpoint, self.check_save_path.format(self.epoch))
        #     print("epoch:", self.epoch)
        #     if self.epoch % 10 == 0 :
        #         torch.save(self.PSNmodel.state_dict(), self.model_save_path.format(self.epoch))
        #     self.epoch += 1
        # wait until cache for sum is reset
        self.n_sum_round = self.n_sum_round + 1
        while self.n_sum_round % self.num_clients != 0:
            time.sleep(self.sleep_time)

        return response

    def MiddleBackward(self, request, context):
        print("middle1")
        # decrypt the vector form cl
        enc = request.msg
        enc_vector = ts.ckks_vector_from(self.ctx, enc)
        dec_vector = enc_vector.decrypt()
        dec_vector = [round(i, 4) for i in dec_vector]
        client_rank = request.client_rank
        ingrad = torch.tensor(dec_vector).cuda()
        rows = int(ingrad.size(0) / 25)
        ingrad = ingrad.unflatten(0, (rows, 25))

        outgrad = backwardComPSN(ingrad, self.optimizer, self.middle_out, self.features)
        num_list = [13, 12]
        self.grad_list = splitGrad(outgrad.cpu(), num_list)
        # print(self.grad_list)
        flag = 0
        if self.grad_list != []:
            flag = 1

        response = tenseal_psndata_pb2.sendgradflag(flag=flag)
        return response

    def BottomBackward(self, request, context):

        self.n_sum_request += 1

        # wait until receiving of all clients' requests
        print("here1")
        while self.n_sum_request % self.num_clients != 0:
            time.sleep(self.sleep_time)
        print("here2")


        client_rank = request.client_rank
        rnd = request.round

        plain_vector = ts.plain_tensor(self.grad_list[client_rank].flatten())
        encrypted_vector = ts.ckks_vector(self.ctx, plain_vector)
        serialize_msg = encrypted_vector.serialize()

        response = tenseal_psndata_pb2.encrypted(round=rnd, client_rank=client_rank, msg=serialize_msg)
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
    tenseal_psndata_pb2_grpc.add_SafeTransmissionServicer_to_server(DataTransServer(1, 2, ctx_file), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("grpc server start...")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
