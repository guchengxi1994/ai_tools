// This file is automatically generated, so please do not edit it.
// @generated by `flutter_rust_bridge`@ 2.5.1.

// ignore_for_file: unused_import, unused_element, unnecessary_import, duplicate_ignore, invalid_use_of_internal_member, annotate_overrides, non_constant_identifier_names, curly_braces_in_flow_control_structures, prefer_const_literals_to_create_immutables, unused_field

import 'api/cv.dart';
import 'api/llm.dart';
import 'api/simple.dart';
import 'dart:async';
import 'dart:convert';
import 'frb_generated.dart';
import 'frb_generated.io.dart'
    if (dart.library.js_interop) 'frb_generated.web.dart';
import 'llm.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';

/// Main entrypoint of the Rust API
class RustLib extends BaseEntrypoint<RustLibApi, RustLibApiImpl, RustLibWire> {
  @internal
  static final instance = RustLib._();

  RustLib._();

  /// Initialize flutter_rust_bridge
  static Future<void> init({
    RustLibApi? api,
    BaseHandler? handler,
    ExternalLibrary? externalLibrary,
  }) async {
    await instance.initImpl(
      api: api,
      handler: handler,
      externalLibrary: externalLibrary,
    );
  }

  /// Initialize flutter_rust_bridge in mock mode.
  /// No libraries for FFI are loaded.
  static void initMock({
    required RustLibApi api,
  }) {
    instance.initMockImpl(
      api: api,
    );
  }

  /// Dispose flutter_rust_bridge
  ///
  /// The call to this function is optional, since flutter_rust_bridge (and everything else)
  /// is automatically disposed when the app stops.
  static void dispose() => instance.disposeImpl();

  @override
  ApiImplConstructor<RustLibApiImpl, RustLibWire> get apiImplConstructor =>
      RustLibApiImpl.new;

  @override
  WireConstructor<RustLibWire> get wireConstructor =>
      RustLibWire.fromExternalLibrary;

  @override
  Future<void> executeRustInitializers() async {
    await api.crateApiSimpleInitApp();
  }

  @override
  ExternalLibraryLoaderConfig get defaultExternalLibraryLoaderConfig =>
      kDefaultExternalLibraryLoaderConfig;

  @override
  String get codegenVersion => '2.5.1';

  @override
  int get rustContentHash => -1687843549;

  static const kDefaultExternalLibraryLoaderConfig =
      ExternalLibraryLoaderConfig(
    stem: 'rust_lib_ai_tools',
    ioDirectory: 'rust/target/release/',
    webPrefix: 'pkg/',
  );
}

abstract class RustLibApi extends BaseApi {
  Future<String> crateApiCvClassifyImage(
      {required String s, required String modelPath, List<String>? classes});

  Stream<ChatResponse> crateApiLlmChatResponseStream();

  Future<void> crateApiLlmClearAllModels();

  String crateApiLlmFormatPrompt(
      {required ChatMessages messages, String? system});

  String crateApiLlmFormatPromptWithThoughtChain(
      {required ChatMessages messages});

  Future<void> crateApiLlmQwen2Chat(
      {required String userPrompt,
      String? systemPrompt,
      required String modelPath});

  Future<void> crateApiLlmQwen2PromptChat(
      {required String prompt, required String modelPath});

  String crateApiSimpleGreet({required String name});

  Future<void> crateApiSimpleInitApp();
}

class RustLibApiImpl extends RustLibApiImplPlatform implements RustLibApi {
  RustLibApiImpl({
    required super.handler,
    required super.wire,
    required super.generalizedFrbRustBinding,
    required super.portManager,
  });

  @override
  Future<String> crateApiCvClassifyImage(
      {required String s, required String modelPath, List<String>? classes}) {
    return handler.executeNormal(NormalTask(
      callFfi: (port_) {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        sse_encode_String(s, serializer);
        sse_encode_String(modelPath, serializer);
        sse_encode_opt_list_String(classes, serializer);
        pdeCallFfi(generalizedFrbRustBinding, serializer,
            funcId: 1, port: port_);
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_String,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiCvClassifyImageConstMeta,
      argValues: [s, modelPath, classes],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiCvClassifyImageConstMeta => const TaskConstMeta(
        debugName: "classify_image",
        argNames: ["s", "modelPath", "classes"],
      );

  @override
  Stream<ChatResponse> crateApiLlmChatResponseStream() {
    final s = RustStreamSink<ChatResponse>();
    handler.executeSync(SyncTask(
      callFfi: () {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        sse_encode_StreamSink_chat_response_Sse(s, serializer);
        return pdeCallFfi(generalizedFrbRustBinding, serializer, funcId: 2)!;
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_unit,
        decodeErrorData: sse_decode_AnyhowException,
      ),
      constMeta: kCrateApiLlmChatResponseStreamConstMeta,
      argValues: [s],
      apiImpl: this,
    ));
    return s.stream;
  }

  TaskConstMeta get kCrateApiLlmChatResponseStreamConstMeta =>
      const TaskConstMeta(
        debugName: "chat_response_stream",
        argNames: ["s"],
      );

  @override
  Future<void> crateApiLlmClearAllModels() {
    return handler.executeNormal(NormalTask(
      callFfi: (port_) {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        pdeCallFfi(generalizedFrbRustBinding, serializer,
            funcId: 3, port: port_);
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_unit,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiLlmClearAllModelsConstMeta,
      argValues: [],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiLlmClearAllModelsConstMeta => const TaskConstMeta(
        debugName: "clear_all_models",
        argNames: [],
      );

  @override
  String crateApiLlmFormatPrompt(
      {required ChatMessages messages, String? system}) {
    return handler.executeSync(SyncTask(
      callFfi: () {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        sse_encode_box_autoadd_chat_messages(messages, serializer);
        sse_encode_opt_String(system, serializer);
        return pdeCallFfi(generalizedFrbRustBinding, serializer, funcId: 4)!;
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_String,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiLlmFormatPromptConstMeta,
      argValues: [messages, system],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiLlmFormatPromptConstMeta => const TaskConstMeta(
        debugName: "format_prompt",
        argNames: ["messages", "system"],
      );

  @override
  String crateApiLlmFormatPromptWithThoughtChain(
      {required ChatMessages messages}) {
    return handler.executeSync(SyncTask(
      callFfi: () {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        sse_encode_box_autoadd_chat_messages(messages, serializer);
        return pdeCallFfi(generalizedFrbRustBinding, serializer, funcId: 5)!;
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_String,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiLlmFormatPromptWithThoughtChainConstMeta,
      argValues: [messages],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiLlmFormatPromptWithThoughtChainConstMeta =>
      const TaskConstMeta(
        debugName: "format_prompt_with_thought_chain",
        argNames: ["messages"],
      );

  @override
  Future<void> crateApiLlmQwen2Chat(
      {required String userPrompt,
      String? systemPrompt,
      required String modelPath}) {
    return handler.executeNormal(NormalTask(
      callFfi: (port_) {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        sse_encode_String(userPrompt, serializer);
        sse_encode_opt_String(systemPrompt, serializer);
        sse_encode_String(modelPath, serializer);
        pdeCallFfi(generalizedFrbRustBinding, serializer,
            funcId: 6, port: port_);
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_unit,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiLlmQwen2ChatConstMeta,
      argValues: [userPrompt, systemPrompt, modelPath],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiLlmQwen2ChatConstMeta => const TaskConstMeta(
        debugName: "qwen2_chat",
        argNames: ["userPrompt", "systemPrompt", "modelPath"],
      );

  @override
  Future<void> crateApiLlmQwen2PromptChat(
      {required String prompt, required String modelPath}) {
    return handler.executeNormal(NormalTask(
      callFfi: (port_) {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        sse_encode_String(prompt, serializer);
        sse_encode_String(modelPath, serializer);
        pdeCallFfi(generalizedFrbRustBinding, serializer,
            funcId: 7, port: port_);
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_unit,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiLlmQwen2PromptChatConstMeta,
      argValues: [prompt, modelPath],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiLlmQwen2PromptChatConstMeta => const TaskConstMeta(
        debugName: "qwen2_prompt_chat",
        argNames: ["prompt", "modelPath"],
      );

  @override
  String crateApiSimpleGreet({required String name}) {
    return handler.executeSync(SyncTask(
      callFfi: () {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        sse_encode_String(name, serializer);
        return pdeCallFfi(generalizedFrbRustBinding, serializer, funcId: 8)!;
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_String,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiSimpleGreetConstMeta,
      argValues: [name],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiSimpleGreetConstMeta => const TaskConstMeta(
        debugName: "greet",
        argNames: ["name"],
      );

  @override
  Future<void> crateApiSimpleInitApp() {
    return handler.executeNormal(NormalTask(
      callFfi: (port_) {
        final serializer = SseSerializer(generalizedFrbRustBinding);
        pdeCallFfi(generalizedFrbRustBinding, serializer,
            funcId: 9, port: port_);
      },
      codec: SseCodec(
        decodeSuccessData: sse_decode_unit,
        decodeErrorData: null,
      ),
      constMeta: kCrateApiSimpleInitAppConstMeta,
      argValues: [],
      apiImpl: this,
    ));
  }

  TaskConstMeta get kCrateApiSimpleInitAppConstMeta => const TaskConstMeta(
        debugName: "init_app",
        argNames: [],
      );

  @protected
  AnyhowException dco_decode_AnyhowException(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return AnyhowException(raw as String);
  }

  @protected
  RustStreamSink<ChatResponse> dco_decode_StreamSink_chat_response_Sse(
      dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    throw UnimplementedError();
  }

  @protected
  String dco_decode_String(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return raw as String;
  }

  @protected
  bool dco_decode_bool(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return raw as bool;
  }

  @protected
  ChatMessages dco_decode_box_autoadd_chat_messages(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return dco_decode_chat_messages(raw);
  }

  @protected
  ChatMessage dco_decode_chat_message(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    final arr = raw as List<dynamic>;
    if (arr.length != 2)
      throw Exception('unexpected arr length: expect 2 but see ${arr.length}');
    return ChatMessage(
      role: dco_decode_String(arr[0]),
      content: dco_decode_String(arr[1]),
    );
  }

  @protected
  ChatMessages dco_decode_chat_messages(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    final arr = raw as List<dynamic>;
    if (arr.length != 1)
      throw Exception('unexpected arr length: expect 1 but see ${arr.length}');
    return ChatMessages(
      field0: dco_decode_list_chat_message(arr[0]),
    );
  }

  @protected
  ChatResponse dco_decode_chat_response(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    final arr = raw as List<dynamic>;
    if (arr.length != 6)
      throw Exception('unexpected arr length: expect 6 but see ${arr.length}');
    return ChatResponse(
      content: dco_decode_String(arr[0]),
      done: dco_decode_bool(arr[1]),
      stage: dco_decode_String(arr[2]),
      uuid: dco_decode_String(arr[3]),
      tps: dco_decode_f_64(arr[4]),
      tokenGenerated: dco_decode_usize(arr[5]),
    );
  }

  @protected
  double dco_decode_f_64(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return raw as double;
  }

  @protected
  List<String> dco_decode_list_String(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return (raw as List<dynamic>).map(dco_decode_String).toList();
  }

  @protected
  List<ChatMessage> dco_decode_list_chat_message(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return (raw as List<dynamic>).map(dco_decode_chat_message).toList();
  }

  @protected
  Uint8List dco_decode_list_prim_u_8_strict(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return raw as Uint8List;
  }

  @protected
  String? dco_decode_opt_String(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return raw == null ? null : dco_decode_String(raw);
  }

  @protected
  List<String>? dco_decode_opt_list_String(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return raw == null ? null : dco_decode_list_String(raw);
  }

  @protected
  int dco_decode_u_8(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return raw as int;
  }

  @protected
  void dco_decode_unit(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return;
  }

  @protected
  BigInt dco_decode_usize(dynamic raw) {
    // Codec=Dco (DartCObject based), see doc to use other codecs
    return dcoDecodeU64(raw);
  }

  @protected
  AnyhowException sse_decode_AnyhowException(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    var inner = sse_decode_String(deserializer);
    return AnyhowException(inner);
  }

  @protected
  RustStreamSink<ChatResponse> sse_decode_StreamSink_chat_response_Sse(
      SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    throw UnimplementedError('Unreachable ()');
  }

  @protected
  String sse_decode_String(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    var inner = sse_decode_list_prim_u_8_strict(deserializer);
    return utf8.decoder.convert(inner);
  }

  @protected
  bool sse_decode_bool(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    return deserializer.buffer.getUint8() != 0;
  }

  @protected
  ChatMessages sse_decode_box_autoadd_chat_messages(
      SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    return (sse_decode_chat_messages(deserializer));
  }

  @protected
  ChatMessage sse_decode_chat_message(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    var var_role = sse_decode_String(deserializer);
    var var_content = sse_decode_String(deserializer);
    return ChatMessage(role: var_role, content: var_content);
  }

  @protected
  ChatMessages sse_decode_chat_messages(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    var var_field0 = sse_decode_list_chat_message(deserializer);
    return ChatMessages(field0: var_field0);
  }

  @protected
  ChatResponse sse_decode_chat_response(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    var var_content = sse_decode_String(deserializer);
    var var_done = sse_decode_bool(deserializer);
    var var_stage = sse_decode_String(deserializer);
    var var_uuid = sse_decode_String(deserializer);
    var var_tps = sse_decode_f_64(deserializer);
    var var_tokenGenerated = sse_decode_usize(deserializer);
    return ChatResponse(
        content: var_content,
        done: var_done,
        stage: var_stage,
        uuid: var_uuid,
        tps: var_tps,
        tokenGenerated: var_tokenGenerated);
  }

  @protected
  double sse_decode_f_64(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    return deserializer.buffer.getFloat64();
  }

  @protected
  List<String> sse_decode_list_String(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs

    var len_ = sse_decode_i_32(deserializer);
    var ans_ = <String>[];
    for (var idx_ = 0; idx_ < len_; ++idx_) {
      ans_.add(sse_decode_String(deserializer));
    }
    return ans_;
  }

  @protected
  List<ChatMessage> sse_decode_list_chat_message(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs

    var len_ = sse_decode_i_32(deserializer);
    var ans_ = <ChatMessage>[];
    for (var idx_ = 0; idx_ < len_; ++idx_) {
      ans_.add(sse_decode_chat_message(deserializer));
    }
    return ans_;
  }

  @protected
  Uint8List sse_decode_list_prim_u_8_strict(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    var len_ = sse_decode_i_32(deserializer);
    return deserializer.buffer.getUint8List(len_);
  }

  @protected
  String? sse_decode_opt_String(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs

    if (sse_decode_bool(deserializer)) {
      return (sse_decode_String(deserializer));
    } else {
      return null;
    }
  }

  @protected
  List<String>? sse_decode_opt_list_String(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs

    if (sse_decode_bool(deserializer)) {
      return (sse_decode_list_String(deserializer));
    } else {
      return null;
    }
  }

  @protected
  int sse_decode_u_8(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    return deserializer.buffer.getUint8();
  }

  @protected
  void sse_decode_unit(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
  }

  @protected
  BigInt sse_decode_usize(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    return deserializer.buffer.getBigUint64();
  }

  @protected
  int sse_decode_i_32(SseDeserializer deserializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    return deserializer.buffer.getInt32();
  }

  @protected
  void sse_encode_AnyhowException(
      AnyhowException self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    sse_encode_String(self.message, serializer);
  }

  @protected
  void sse_encode_StreamSink_chat_response_Sse(
      RustStreamSink<ChatResponse> self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    sse_encode_String(
        self.setupAndSerialize(
            codec: SseCodec(
          decodeSuccessData: sse_decode_chat_response,
          decodeErrorData: sse_decode_AnyhowException,
        )),
        serializer);
  }

  @protected
  void sse_encode_String(String self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    sse_encode_list_prim_u_8_strict(utf8.encoder.convert(self), serializer);
  }

  @protected
  void sse_encode_bool(bool self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    serializer.buffer.putUint8(self ? 1 : 0);
  }

  @protected
  void sse_encode_box_autoadd_chat_messages(
      ChatMessages self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    sse_encode_chat_messages(self, serializer);
  }

  @protected
  void sse_encode_chat_message(ChatMessage self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    sse_encode_String(self.role, serializer);
    sse_encode_String(self.content, serializer);
  }

  @protected
  void sse_encode_chat_messages(ChatMessages self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    sse_encode_list_chat_message(self.field0, serializer);
  }

  @protected
  void sse_encode_chat_response(ChatResponse self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    sse_encode_String(self.content, serializer);
    sse_encode_bool(self.done, serializer);
    sse_encode_String(self.stage, serializer);
    sse_encode_String(self.uuid, serializer);
    sse_encode_f_64(self.tps, serializer);
    sse_encode_usize(self.tokenGenerated, serializer);
  }

  @protected
  void sse_encode_f_64(double self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    serializer.buffer.putFloat64(self);
  }

  @protected
  void sse_encode_list_String(List<String> self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    sse_encode_i_32(self.length, serializer);
    for (final item in self) {
      sse_encode_String(item, serializer);
    }
  }

  @protected
  void sse_encode_list_chat_message(
      List<ChatMessage> self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    sse_encode_i_32(self.length, serializer);
    for (final item in self) {
      sse_encode_chat_message(item, serializer);
    }
  }

  @protected
  void sse_encode_list_prim_u_8_strict(
      Uint8List self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    sse_encode_i_32(self.length, serializer);
    serializer.buffer.putUint8List(self);
  }

  @protected
  void sse_encode_opt_String(String? self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs

    sse_encode_bool(self != null, serializer);
    if (self != null) {
      sse_encode_String(self, serializer);
    }
  }

  @protected
  void sse_encode_opt_list_String(
      List<String>? self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs

    sse_encode_bool(self != null, serializer);
    if (self != null) {
      sse_encode_list_String(self, serializer);
    }
  }

  @protected
  void sse_encode_u_8(int self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    serializer.buffer.putUint8(self);
  }

  @protected
  void sse_encode_unit(void self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
  }

  @protected
  void sse_encode_usize(BigInt self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    serializer.buffer.putBigUint64(self);
  }

  @protected
  void sse_encode_i_32(int self, SseSerializer serializer) {
    // Codec=Sse (Serialization based), see doc to use other codecs
    serializer.buffer.putInt32(self);
  }
}
