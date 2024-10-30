// This file is automatically generated, so please do not edit it.
// @generated by `flutter_rust_bridge`@ 2.5.1.

// ignore_for_file: invalid_use_of_internal_member, unused_import, unnecessary_import

import '../frb_generated.dart';
import '../llm.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';

Future<void> qwen2Chat(
        {required String userPrompt,
        String? systemPrompt,
        required String modelPath}) =>
    RustLib.instance.api.crateApiLlmQwen2Chat(
        userPrompt: userPrompt,
        systemPrompt: systemPrompt,
        modelPath: modelPath);

Stream<ChatResponse> chatResponseStream() =>
    RustLib.instance.api.crateApiLlmChatResponseStream();
