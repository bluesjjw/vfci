# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import tenseal_DragonNetData_pb2 as tenseal__DragonNetData__pb2


class SafeTransmissionStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.MiddleForward = channel.unary_unary(
                '/SafeTransmission/MiddleForward',
                request_serializer=tenseal__DragonNetData__pb2.encrypted.SerializeToString,
                response_deserializer=tenseal__DragonNetData__pb2.encrypted.FromString,
                )
        self.MiddleBackward = channel.unary_unary(
                '/SafeTransmission/MiddleBackward',
                request_serializer=tenseal__DragonNetData__pb2.gradenc.SerializeToString,
                response_deserializer=tenseal__DragonNetData__pb2.sendgradflag.FromString,
                )
        self.BottomBackward = channel.unary_unary(
                '/SafeTransmission/BottomBackward',
                request_serializer=tenseal__DragonNetData__pb2.sendgrad.SerializeToString,
                response_deserializer=tenseal__DragonNetData__pb2.encrypted.FromString,
                )


class SafeTransmissionServicer(object):
    """Missing associated documentation comment in .proto file."""

    def MiddleForward(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def MiddleBackward(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def BottomBackward(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SafeTransmissionServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'MiddleForward': grpc.unary_unary_rpc_method_handler(
                    servicer.MiddleForward,
                    request_deserializer=tenseal__DragonNetData__pb2.encrypted.FromString,
                    response_serializer=tenseal__DragonNetData__pb2.encrypted.SerializeToString,
            ),
            'MiddleBackward': grpc.unary_unary_rpc_method_handler(
                    servicer.MiddleBackward,
                    request_deserializer=tenseal__DragonNetData__pb2.gradenc.FromString,
                    response_serializer=tenseal__DragonNetData__pb2.sendgradflag.SerializeToString,
            ),
            'BottomBackward': grpc.unary_unary_rpc_method_handler(
                    servicer.BottomBackward,
                    request_deserializer=tenseal__DragonNetData__pb2.sendgrad.FromString,
                    response_serializer=tenseal__DragonNetData__pb2.encrypted.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'SafeTransmission', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class SafeTransmission(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def MiddleForward(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/SafeTransmission/MiddleForward',
            tenseal__DragonNetData__pb2.encrypted.SerializeToString,
            tenseal__DragonNetData__pb2.encrypted.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def MiddleBackward(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/SafeTransmission/MiddleBackward',
            tenseal__DragonNetData__pb2.gradenc.SerializeToString,
            tenseal__DragonNetData__pb2.sendgradflag.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def BottomBackward(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/SafeTransmission/BottomBackward',
            tenseal__DragonNetData__pb2.sendgrad.SerializeToString,
            tenseal__DragonNetData__pb2.encrypted.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
