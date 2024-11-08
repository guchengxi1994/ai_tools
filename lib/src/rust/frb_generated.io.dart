// This file is automatically generated, so please do not edit it.
// @generated by `flutter_rust_bridge`@ 2.5.1.

// ignore_for_file: unused_import, unused_element, unnecessary_import, duplicate_ignore, invalid_use_of_internal_member, annotate_overrides, non_constant_identifier_names, curly_braces_in_flow_control_structures, prefer_const_literals_to_create_immutables, unused_field

import 'api/cv.dart';
import 'api/llm.dart';
import 'api/simple.dart';
import 'api/tools.dart';
import 'cv/object_detect_result.dart';
import 'dart:async';
import 'dart:convert';
import 'dart:ffi' as ffi;
import 'frb_generated.dart';
import 'llm.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated_io.dart';
import 'tools.dart';

abstract class RustLibApiImplPlatform extends BaseApiImpl<RustLibWire> {
  RustLibApiImplPlatform({
    required super.handler,
    required super.wire,
    required super.generalizedFrbRustBinding,
    required super.portManager,
  });

  @protected
  AnyhowException dco_decode_AnyhowException(dynamic raw);

  @protected
  RustStreamSink<String> dco_decode_StreamSink_String_Sse(dynamic raw);

  @protected
  RustStreamSink<ChatResponse> dco_decode_StreamSink_chat_response_Sse(
      dynamic raw);

  @protected
  RustStreamSink<TrainMessage> dco_decode_StreamSink_train_message_Sse(
      dynamic raw);

  @protected
  String dco_decode_String(dynamic raw);

  @protected
  bool dco_decode_bool(dynamic raw);

  @protected
  ChatMessages dco_decode_box_autoadd_chat_messages(dynamic raw);

  @protected
  BigInt dco_decode_box_autoadd_usize(dynamic raw);

  @protected
  ChatMessage dco_decode_chat_message(dynamic raw);

  @protected
  ChatMessages dco_decode_chat_messages(dynamic raw);

  @protected
  ChatResponse dco_decode_chat_response(dynamic raw);

  @protected
  double dco_decode_f_32(dynamic raw);

  @protected
  double dco_decode_f_64(dynamic raw);

  @protected
  int dco_decode_i_32(dynamic raw);

  @protected
  List<String> dco_decode_list_String(dynamic raw);

  @protected
  List<ChatMessage> dco_decode_list_chat_message(dynamic raw);

  @protected
  List<ObjectDetectResult> dco_decode_list_object_detect_result(dynamic raw);

  @protected
  List<int> dco_decode_list_prim_u_8_loose(dynamic raw);

  @protected
  Uint8List dco_decode_list_prim_u_8_strict(dynamic raw);

  @protected
  ObjectDetectResult dco_decode_object_detect_result(dynamic raw);

  @protected
  String? dco_decode_opt_String(dynamic raw);

  @protected
  BigInt? dco_decode_opt_box_autoadd_usize(dynamic raw);

  @protected
  List<String>? dco_decode_opt_list_String(dynamic raw);

  @protected
  TrainMessage dco_decode_train_message(dynamic raw);

  @protected
  int dco_decode_u_8(dynamic raw);

  @protected
  void dco_decode_unit(dynamic raw);

  @protected
  BigInt dco_decode_usize(dynamic raw);

  @protected
  AnyhowException sse_decode_AnyhowException(SseDeserializer deserializer);

  @protected
  RustStreamSink<String> sse_decode_StreamSink_String_Sse(
      SseDeserializer deserializer);

  @protected
  RustStreamSink<ChatResponse> sse_decode_StreamSink_chat_response_Sse(
      SseDeserializer deserializer);

  @protected
  RustStreamSink<TrainMessage> sse_decode_StreamSink_train_message_Sse(
      SseDeserializer deserializer);

  @protected
  String sse_decode_String(SseDeserializer deserializer);

  @protected
  bool sse_decode_bool(SseDeserializer deserializer);

  @protected
  ChatMessages sse_decode_box_autoadd_chat_messages(
      SseDeserializer deserializer);

  @protected
  BigInt sse_decode_box_autoadd_usize(SseDeserializer deserializer);

  @protected
  ChatMessage sse_decode_chat_message(SseDeserializer deserializer);

  @protected
  ChatMessages sse_decode_chat_messages(SseDeserializer deserializer);

  @protected
  ChatResponse sse_decode_chat_response(SseDeserializer deserializer);

  @protected
  double sse_decode_f_32(SseDeserializer deserializer);

  @protected
  double sse_decode_f_64(SseDeserializer deserializer);

  @protected
  int sse_decode_i_32(SseDeserializer deserializer);

  @protected
  List<String> sse_decode_list_String(SseDeserializer deserializer);

  @protected
  List<ChatMessage> sse_decode_list_chat_message(SseDeserializer deserializer);

  @protected
  List<ObjectDetectResult> sse_decode_list_object_detect_result(
      SseDeserializer deserializer);

  @protected
  List<int> sse_decode_list_prim_u_8_loose(SseDeserializer deserializer);

  @protected
  Uint8List sse_decode_list_prim_u_8_strict(SseDeserializer deserializer);

  @protected
  ObjectDetectResult sse_decode_object_detect_result(
      SseDeserializer deserializer);

  @protected
  String? sse_decode_opt_String(SseDeserializer deserializer);

  @protected
  BigInt? sse_decode_opt_box_autoadd_usize(SseDeserializer deserializer);

  @protected
  List<String>? sse_decode_opt_list_String(SseDeserializer deserializer);

  @protected
  TrainMessage sse_decode_train_message(SseDeserializer deserializer);

  @protected
  int sse_decode_u_8(SseDeserializer deserializer);

  @protected
  void sse_decode_unit(SseDeserializer deserializer);

  @protected
  BigInt sse_decode_usize(SseDeserializer deserializer);

  @protected
  void sse_encode_AnyhowException(
      AnyhowException self, SseSerializer serializer);

  @protected
  void sse_encode_StreamSink_String_Sse(
      RustStreamSink<String> self, SseSerializer serializer);

  @protected
  void sse_encode_StreamSink_chat_response_Sse(
      RustStreamSink<ChatResponse> self, SseSerializer serializer);

  @protected
  void sse_encode_StreamSink_train_message_Sse(
      RustStreamSink<TrainMessage> self, SseSerializer serializer);

  @protected
  void sse_encode_String(String self, SseSerializer serializer);

  @protected
  void sse_encode_bool(bool self, SseSerializer serializer);

  @protected
  void sse_encode_box_autoadd_chat_messages(
      ChatMessages self, SseSerializer serializer);

  @protected
  void sse_encode_box_autoadd_usize(BigInt self, SseSerializer serializer);

  @protected
  void sse_encode_chat_message(ChatMessage self, SseSerializer serializer);

  @protected
  void sse_encode_chat_messages(ChatMessages self, SseSerializer serializer);

  @protected
  void sse_encode_chat_response(ChatResponse self, SseSerializer serializer);

  @protected
  void sse_encode_f_32(double self, SseSerializer serializer);

  @protected
  void sse_encode_f_64(double self, SseSerializer serializer);

  @protected
  void sse_encode_i_32(int self, SseSerializer serializer);

  @protected
  void sse_encode_list_String(List<String> self, SseSerializer serializer);

  @protected
  void sse_encode_list_chat_message(
      List<ChatMessage> self, SseSerializer serializer);

  @protected
  void sse_encode_list_object_detect_result(
      List<ObjectDetectResult> self, SseSerializer serializer);

  @protected
  void sse_encode_list_prim_u_8_loose(List<int> self, SseSerializer serializer);

  @protected
  void sse_encode_list_prim_u_8_strict(
      Uint8List self, SseSerializer serializer);

  @protected
  void sse_encode_object_detect_result(
      ObjectDetectResult self, SseSerializer serializer);

  @protected
  void sse_encode_opt_String(String? self, SseSerializer serializer);

  @protected
  void sse_encode_opt_box_autoadd_usize(BigInt? self, SseSerializer serializer);

  @protected
  void sse_encode_opt_list_String(List<String>? self, SseSerializer serializer);

  @protected
  void sse_encode_train_message(TrainMessage self, SseSerializer serializer);

  @protected
  void sse_encode_u_8(int self, SseSerializer serializer);

  @protected
  void sse_encode_unit(void self, SseSerializer serializer);

  @protected
  void sse_encode_usize(BigInt self, SseSerializer serializer);
}

// Section: wire_class

class RustLibWire implements BaseWire {
  factory RustLibWire.fromExternalLibrary(ExternalLibrary lib) =>
      RustLibWire(lib.ffiDynamicLibrary);

  /// Holds the symbol lookup function.
  final ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName)
      _lookup;

  /// The symbols are looked up in [dynamicLibrary].
  RustLibWire(ffi.DynamicLibrary dynamicLibrary)
      : _lookup = dynamicLibrary.lookup;
}
