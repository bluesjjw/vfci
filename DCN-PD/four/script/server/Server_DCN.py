from concurrent import futures
import sys

sys.path.append("..")
sys.path.append("../..")
import torch
import torch.optim as optim
import time
import transmission.tenseal.tenseal_dcndata_pb2 as tenseal_dcndata_pb2
import transmission.tenseal.tenseal_dcndata_pb2_grpc as tenseal_dcndata_pb2_grpc
import tenseal as ts
import grpc
from transmission.Utils import forwardMiddle, backwardMiddleDCN
from models.MiddlePSN import MiddlePSN
from models.MiddleDCN import MiddleDCN

from tkinter import _flatten


class DataTransServer(tenseal_dcndata_pb2_grpc.SafeTransmissionServicer):

    def __init__(self, address, num_clients, ctx_file):
        self.address = address
        self.num_clients = num_clients
        self.ctx_file = ctx_file
        context_bytes = open(self.ctx_file, "rb").read()
        self.ctx = ts.context_from(context_bytes)
        self.sleep_time = 0.1
        self.PSNmodel = MiddlePSN.cuda()
        self.PSNmodel.eval()
        self.PSNmodel.load_state_dict(
            torch.load(''))
        self.middleDCN = MiddleDCN(training_flag=True).cuda()
        self.optimizer = optim.Adam(self.middleDCN.parameters(), lr=0.001)
        self.features = None
        self.prob = None
        self.middleOut = None
        self.model_save_path = ''
        self.check_save_path = ''

        # cache and counter for sum operation
        self.n_sum_round = 0
        self.decdict = {}
        self.vectors_list = []
        self.sum_data = []
        self.n_sum_request = 0
        self.n_sum_response = 0
        self.getITE_completed = False
        print("server lauched")

    def reset_sum(self):
        self.vectors_list.clear()
        self.sum_data = []
        self.n_sum_request = 0
        self.n_sum_response = 0
        self.decdict.clear()
        self.getITE_completed = False

    def sortResdict(self):
        sortlist = sorted(self.decdict)
        for i in sortlist:
            self.vectors_list.append(self.decdict[i])

    def PscoreTransmit(self, request, context):
        rnd = request.round
        epoch = request.epoch
        client_rank = request.client_rank
        enc = request.msg
        enc_vector = ts.ckks_vector_from(self.ctx, enc)
        dec_vector = enc_vector.decrypt()

        self.prob = torch.tensor(dec_vector)
        # print("prob ", self.prob)
        self.middleOut = forwardMiddle(self.features, self.prob, self.middleDCN)

        # encrypt the predicted ITE
        plain_vector = ts.plain_tensor(self.middleOut.detach().cpu())
        encrypted_vector = ts.ckks_vector(self.ctx, plain_vector)
        serialize_msg = encrypted_vector.serialize()

        # create response and wait until all response is completed
        response = tenseal_dcndata_pb2.gradenc(client_rank=client_rank, msg=serialize_msg)

        if rnd % 111 == 0 or rnd % 486 == 0:
            print(self.features.shape)
            checkpoint = {'optimizer': self.optimizer.state_dict(), 'net': self.middleDCN.state_dict()}
            torch.save(checkpoint, self.check_save_path.format(epoch))
        if epoch % 10 == 0:
            torch.save(self.middleDCN.state_dict(), self.model_save_path.format(epoch))

        return response

    def MiddleForward(self, request, context):
        # decrypt the vector form client
        enc = request.msg
        enc_vector = ts.ckks_vector_from(self.ctx, enc)
        dec_vector = enc_vector.decrypt()
        # dec_vector = [round(i, 4) for i in dec_vector]
        client_rank = request.client_rank

        epoch = request.epoch
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
            print("epoch ", epoch)
            features = list(_flatten(self.vectors_list))
            self.features = torch.tensor(features, requires_grad=True).cuda()



        plain_vector = ts.plain_tensor(self.features.detach().cpu())
        encrypted_vector = ts.ckks_vector(self.ctx, plain_vector)
        serialize_msg = encrypted_vector.serialize()

        # create response and wait until all response is completed
        response = tenseal_dcndata_pb2.gradenc(client_rank=client_rank, msg=serialize_msg)
        self.n_sum_response = self.n_sum_response + 1
        while self.n_sum_response % self.num_clients != 0:
            time.sleep(self.sleep_time)

        # reset the sum
        if client_rank == self.num_clients - 1:
            self.reset_sum()

        # wait until cache for sum is reset
        self.n_sum_round = self.n_sum_round + 1
        while self.n_sum_round % self.num_clients != 0:
            time.sleep(self.sleep_time)

        return response

    def MiddleBackward(self, request, context):

        # decrypt the vector form label owner
        enc = request.msg
        enc_vector = ts.ckks_vector_from(self.ctx, enc)
        dec_vector = enc_vector.decrypt()

        ingrad = torch.tensor(dec_vector).cuda()

        # print("here ",ingrad)
        backwardMiddleDCN(ingrad, self.ITE, self.optimizer)

        response = tenseal_dcndata_pb2.sendgradflag(flag=1)
        return response


def serve():
    max_msg_size = 1000000000
    ctx_file = "../../transmission/tenseal/ts_ckks_tiny.config"
    options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5), options=options)
    tenseal_dcndata_pb2_grpc.add_SafeTransmissionServicer_to_server(DataTransServer(1, 2, ctx_file), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("grpc server start...")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
