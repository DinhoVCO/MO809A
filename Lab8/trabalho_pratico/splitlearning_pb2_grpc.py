# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import splitlearning_pb2 as splitlearning__pb2

GRPC_GENERATED_VERSION = '1.66.2'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in splitlearning_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class SplitLearningStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.SendActivation = channel.unary_unary(
                '/splitlearning.SplitLearning/SendActivation',
                request_serializer=splitlearning__pb2.ActivationsRequest.SerializeToString,
                response_deserializer=splitlearning__pb2.ActivationsResponse.FromString,
                _registered_method=True)
        self.SendGradient = channel.unary_unary(
                '/splitlearning.SplitLearning/SendGradient',
                request_serializer=splitlearning__pb2.GradientsRequest.SerializeToString,
                response_deserializer=splitlearning__pb2.GradientsResponse.FromString,
                _registered_method=True)


class SplitLearningServicer(object):
    """Missing associated documentation comment in .proto file."""

    def SendActivation(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendGradient(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SplitLearningServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'SendActivation': grpc.unary_unary_rpc_method_handler(
                    servicer.SendActivation,
                    request_deserializer=splitlearning__pb2.ActivationsRequest.FromString,
                    response_serializer=splitlearning__pb2.ActivationsResponse.SerializeToString,
            ),
            'SendGradient': grpc.unary_unary_rpc_method_handler(
                    servicer.SendGradient,
                    request_deserializer=splitlearning__pb2.GradientsRequest.FromString,
                    response_serializer=splitlearning__pb2.GradientsResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'splitlearning.SplitLearning', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('splitlearning.SplitLearning', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class SplitLearning(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def SendActivation(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/splitlearning.SplitLearning/SendActivation',
            splitlearning__pb2.ActivationsRequest.SerializeToString,
            splitlearning__pb2.ActivationsResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def SendGradient(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/splitlearning.SplitLearning/SendGradient',
            splitlearning__pb2.GradientsRequest.SerializeToString,
            splitlearning__pb2.GradientsResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
