// This file is automatically generated, so please do not edit it.
// Generated by `flutter_rust_bridge`@ 2.2.0.

// ignore_for_file: unused_import, unused_element, unnecessary_import, duplicate_ignore, invalid_use_of_internal_member, annotate_overrides, non_constant_identifier_names, curly_braces_in_flow_control_structures, prefer_const_literals_to_create_immutables, unused_field

import 'api/cv.dart';
import 'api/llm.dart';
import 'api/simple.dart';
import 'dart:async';
import 'dart:convert';
import 'frb_generated.dart';
import 'frb_generated.io.dart' if (dart.library.js_interop) 'frb_generated.web.dart';
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

                  /// Dispose flutter_rust_bridge
                  ///
                  /// The call to this function is optional, since flutter_rust_bridge (and everything else)
                  /// is automatically disposed when the app stops.
                  static void dispose() => instance.disposeImpl();

                  @override
                  ApiImplConstructor<RustLibApiImpl, RustLibWire> get apiImplConstructor => RustLibApiImpl.new;

                  @override
                  WireConstructor<RustLibWire> get wireConstructor => RustLibWire.fromExternalLibrary;

                  @override
                  Future<void> executeRustInitializers() async {
                    await api.crateApiSimpleInitApp();

                  }

                  @override
                  ExternalLibraryLoaderConfig get defaultExternalLibraryLoaderConfig => kDefaultExternalLibraryLoaderConfig;

                  @override
                  String get codegenVersion => '2.2.0';

                  @override
                  int get rustContentHash => -1305452514;

                  static const kDefaultExternalLibraryLoaderConfig = ExternalLibraryLoaderConfig(
                    stem: 'rust_lib_ai_tools',
                    ioDirectory: 'rust/target/release/',
                    webPrefix: 'pkg/',
                  );
                }
                

                abstract class RustLibApi extends BaseApi {
                  Future<String> crateApiCvClassifyImage({required String s , required String modelPath , List<String>? classes });

Future<void> crateApiLlmQwen2Chat({required String userPrompt , String? systemPrompt , required String modelPath });

String crateApiSimpleGreet({required String name });

Future<void> crateApiSimpleInitApp();


                }
                

                class RustLibApiImpl extends RustLibApiImplPlatform implements RustLibApi {
                  RustLibApiImpl({
                    required super.handler,
                    required super.wire,
                    required super.generalizedFrbRustBinding,
                    required super.portManager,
                  });

                  @override Future<String> crateApiCvClassifyImage({required String s , required String modelPath , List<String>? classes })  { return handler.executeNormal(NormalTask(
            callFfi: (port_) {
              
            final serializer = SseSerializer(generalizedFrbRustBinding);sse_encode_String(s, serializer);
sse_encode_String(modelPath, serializer);
sse_encode_opt_list_String(classes, serializer);
            pdeCallFfi(generalizedFrbRustBinding, serializer, funcId: 1, port: port_);
            
            },
            codec: 
        SseCodec(
          decodeSuccessData: sse_decode_String,
          decodeErrorData: null,
        )
        ,
            constMeta: kCrateApiCvClassifyImageConstMeta,
            argValues: [s, modelPath, classes],
            apiImpl: this,
        )); }


        TaskConstMeta get kCrateApiCvClassifyImageConstMeta => const TaskConstMeta(
            debugName: "classify_image",
            argNames: ["s", "modelPath", "classes"],
        );
        

@override Future<void> crateApiLlmQwen2Chat({required String userPrompt , String? systemPrompt , required String modelPath })  { return handler.executeNormal(NormalTask(
            callFfi: (port_) {
              
            final serializer = SseSerializer(generalizedFrbRustBinding);sse_encode_String(userPrompt, serializer);
sse_encode_opt_String(systemPrompt, serializer);
sse_encode_String(modelPath, serializer);
            pdeCallFfi(generalizedFrbRustBinding, serializer, funcId: 2, port: port_);
            
            },
            codec: 
        SseCodec(
          decodeSuccessData: sse_decode_unit,
          decodeErrorData: null,
        )
        ,
            constMeta: kCrateApiLlmQwen2ChatConstMeta,
            argValues: [userPrompt, systemPrompt, modelPath],
            apiImpl: this,
        )); }


        TaskConstMeta get kCrateApiLlmQwen2ChatConstMeta => const TaskConstMeta(
            debugName: "qwen2_chat",
            argNames: ["userPrompt", "systemPrompt", "modelPath"],
        );
        

@override String crateApiSimpleGreet({required String name })  { return handler.executeSync(SyncTask(
            callFfi: () {
              
            final serializer = SseSerializer(generalizedFrbRustBinding);sse_encode_String(name, serializer);
            return pdeCallFfi(generalizedFrbRustBinding, serializer, funcId: 3)!;
            
            },
            codec: 
        SseCodec(
          decodeSuccessData: sse_decode_String,
          decodeErrorData: null,
        )
        ,
            constMeta: kCrateApiSimpleGreetConstMeta,
            argValues: [name],
            apiImpl: this,
        )); }


        TaskConstMeta get kCrateApiSimpleGreetConstMeta => const TaskConstMeta(
            debugName: "greet",
            argNames: ["name"],
        );
        

@override Future<void> crateApiSimpleInitApp()  { return handler.executeNormal(NormalTask(
            callFfi: (port_) {
              
            final serializer = SseSerializer(generalizedFrbRustBinding);
            pdeCallFfi(generalizedFrbRustBinding, serializer, funcId: 4, port: port_);
            
            },
            codec: 
        SseCodec(
          decodeSuccessData: sse_decode_unit,
          decodeErrorData: null,
        )
        ,
            constMeta: kCrateApiSimpleInitAppConstMeta,
            argValues: [],
            apiImpl: this,
        )); }


        TaskConstMeta get kCrateApiSimpleInitAppConstMeta => const TaskConstMeta(
            debugName: "init_app",
            argNames: [],
        );
        



                  @protected String dco_decode_String(dynamic raw){ // Codec=Dco (DartCObject based), see doc to use other codecs
return raw as String; }

@protected List<String> dco_decode_list_String(dynamic raw){ // Codec=Dco (DartCObject based), see doc to use other codecs
return (raw as List<dynamic>).map(dco_decode_String).toList(); }

@protected Uint8List dco_decode_list_prim_u_8_strict(dynamic raw){ // Codec=Dco (DartCObject based), see doc to use other codecs
return raw as Uint8List; }

@protected String? dco_decode_opt_String(dynamic raw){ // Codec=Dco (DartCObject based), see doc to use other codecs
return raw == null ? null : dco_decode_String(raw); }

@protected List<String>? dco_decode_opt_list_String(dynamic raw){ // Codec=Dco (DartCObject based), see doc to use other codecs
return raw == null ? null : dco_decode_list_String(raw); }

@protected int dco_decode_u_8(dynamic raw){ // Codec=Dco (DartCObject based), see doc to use other codecs
return raw as int; }

@protected void dco_decode_unit(dynamic raw){ // Codec=Dco (DartCObject based), see doc to use other codecs
return; }

@protected String sse_decode_String(SseDeserializer deserializer){ // Codec=Sse (Serialization based), see doc to use other codecs
var inner = sse_decode_list_prim_u_8_strict(deserializer);
        return utf8.decoder.convert(inner); }

@protected List<String> sse_decode_list_String(SseDeserializer deserializer){ // Codec=Sse (Serialization based), see doc to use other codecs

        var len_ = sse_decode_i_32(deserializer);
        var ans_ = <String>[];
        for (var idx_ = 0; idx_ < len_; ++idx_) { ans_.add(sse_decode_String(deserializer)); }
        return ans_;
         }

@protected Uint8List sse_decode_list_prim_u_8_strict(SseDeserializer deserializer){ // Codec=Sse (Serialization based), see doc to use other codecs
var len_ = sse_decode_i_32(deserializer);
                return deserializer.buffer.getUint8List(len_); }

@protected String? sse_decode_opt_String(SseDeserializer deserializer){ // Codec=Sse (Serialization based), see doc to use other codecs

            if (sse_decode_bool(deserializer)) {
                return (sse_decode_String(deserializer));
            } else {
                return null;
            }
             }

@protected List<String>? sse_decode_opt_list_String(SseDeserializer deserializer){ // Codec=Sse (Serialization based), see doc to use other codecs

            if (sse_decode_bool(deserializer)) {
                return (sse_decode_list_String(deserializer));
            } else {
                return null;
            }
             }

@protected int sse_decode_u_8(SseDeserializer deserializer){ // Codec=Sse (Serialization based), see doc to use other codecs
return deserializer.buffer.getUint8(); }

@protected void sse_decode_unit(SseDeserializer deserializer){ // Codec=Sse (Serialization based), see doc to use other codecs
 }

@protected int sse_decode_i_32(SseDeserializer deserializer){ // Codec=Sse (Serialization based), see doc to use other codecs
return deserializer.buffer.getInt32(); }

@protected bool sse_decode_bool(SseDeserializer deserializer){ // Codec=Sse (Serialization based), see doc to use other codecs
return deserializer.buffer.getUint8() != 0; }

@protected void sse_encode_String(String self, SseSerializer serializer){ // Codec=Sse (Serialization based), see doc to use other codecs
sse_encode_list_prim_u_8_strict(utf8.encoder.convert(self), serializer); }

@protected void sse_encode_list_String(List<String> self, SseSerializer serializer){ // Codec=Sse (Serialization based), see doc to use other codecs
sse_encode_i_32(self.length, serializer);
        for (final item in self) { sse_encode_String(item, serializer); } }

@protected void sse_encode_list_prim_u_8_strict(Uint8List self, SseSerializer serializer){ // Codec=Sse (Serialization based), see doc to use other codecs
sse_encode_i_32(self.length, serializer);
                    serializer.buffer.putUint8List(self); }

@protected void sse_encode_opt_String(String? self, SseSerializer serializer){ // Codec=Sse (Serialization based), see doc to use other codecs

                sse_encode_bool(self != null, serializer);
                if (self != null) {
                    sse_encode_String(self, serializer);
                }
                 }

@protected void sse_encode_opt_list_String(List<String>? self, SseSerializer serializer){ // Codec=Sse (Serialization based), see doc to use other codecs

                sse_encode_bool(self != null, serializer);
                if (self != null) {
                    sse_encode_list_String(self, serializer);
                }
                 }

@protected void sse_encode_u_8(int self, SseSerializer serializer){ // Codec=Sse (Serialization based), see doc to use other codecs
serializer.buffer.putUint8(self); }

@protected void sse_encode_unit(void self, SseSerializer serializer){ // Codec=Sse (Serialization based), see doc to use other codecs
 }

@protected void sse_encode_i_32(int self, SseSerializer serializer){ // Codec=Sse (Serialization based), see doc to use other codecs
serializer.buffer.putInt32(self); }

@protected void sse_encode_bool(bool self, SseSerializer serializer){ // Codec=Sse (Serialization based), see doc to use other codecs
serializer.buffer.putUint8(self ? 1 : 0); }
                }
                
