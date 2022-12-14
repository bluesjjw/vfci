# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tenseal_dcndata.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x15tenseal_dcndata.proto\"K\n\tencrypted\x12\r\n\x05round\x18\x01 \x01(\x05\x12\x13\n\x0b\x63lient_rank\x18\x02 \x01(\x05\x12\r\n\x05\x65poch\x18\x03 \x01(\x05\x12\x0b\n\x03msg\x18\x04 \x01(\x0c\"+\n\x07gradenc\x12\x13\n\x0b\x63lient_rank\x18\x01 \x01(\x05\x12\x0b\n\x03msg\x18\x02 \x01(\x0c\"\x1c\n\x0csendgradflag\x12\x0c\n\x04\x66lag\x18\x01 \x01(\x05\x32\x8c\x01\n\x10SafeTransmission\x12%\n\rMiddleForward\x12\n.encrypted\x1a\x08.gradenc\x12)\n\x0eMiddleBackward\x12\x08.gradenc\x1a\r.sendgradflag\x12&\n\x0ePscoreTransmit\x12\n.encrypted\x1a\x08.gradencb\x06proto3')



_ENCRYPTED = DESCRIPTOR.message_types_by_name['encrypted']
_GRADENC = DESCRIPTOR.message_types_by_name['gradenc']
_SENDGRADFLAG = DESCRIPTOR.message_types_by_name['sendgradflag']
encrypted = _reflection.GeneratedProtocolMessageType('encrypted', (_message.Message,), {
  'DESCRIPTOR' : _ENCRYPTED,
  '__module__' : 'tenseal_dcndata_pb2'
  # @@protoc_insertion_point(class_scope:encrypted)
  })
_sym_db.RegisterMessage(encrypted)

gradenc = _reflection.GeneratedProtocolMessageType('gradenc', (_message.Message,), {
  'DESCRIPTOR' : _GRADENC,
  '__module__' : 'tenseal_dcndata_pb2'
  # @@protoc_insertion_point(class_scope:gradenc)
  })
_sym_db.RegisterMessage(gradenc)

sendgradflag = _reflection.GeneratedProtocolMessageType('sendgradflag', (_message.Message,), {
  'DESCRIPTOR' : _SENDGRADFLAG,
  '__module__' : 'tenseal_dcndata_pb2'
  # @@protoc_insertion_point(class_scope:sendgradflag)
  })
_sym_db.RegisterMessage(sendgradflag)

_SAFETRANSMISSION = DESCRIPTOR.services_by_name['SafeTransmission']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _ENCRYPTED._serialized_start=25
  _ENCRYPTED._serialized_end=100
  _GRADENC._serialized_start=102
  _GRADENC._serialized_end=145
  _SENDGRADFLAG._serialized_start=147
  _SENDGRADFLAG._serialized_end=175
  _SAFETRANSMISSION._serialized_start=178
  _SAFETRANSMISSION._serialized_end=318
# @@protoc_insertion_point(module_scope)
