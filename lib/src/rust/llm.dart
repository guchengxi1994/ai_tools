// This file is automatically generated, so please do not edit it.
// @generated by `flutter_rust_bridge`@ 2.5.1.

// ignore_for_file: invalid_use_of_internal_member, unused_import, unnecessary_import

import 'frb_generated.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';

class ChatMessage {
  final String role;
  final String content;

  const ChatMessage({
    required this.role,
    required this.content,
  });

  @override
  int get hashCode => role.hashCode ^ content.hashCode;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ChatMessage &&
          runtimeType == other.runtimeType &&
          role == other.role &&
          content == other.content;
}

class ChatMessages {
  final List<ChatMessage> field0;

  const ChatMessages({
    required this.field0,
  });

  @override
  int get hashCode => field0.hashCode;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ChatMessages &&
          runtimeType == other.runtimeType &&
          field0 == other.field0;
}

class ChatResponse {
  final String content;
  final bool done;
  final String stage;
  final String uuid;
  final double tps;
  final BigInt tokenGenerated;

  const ChatResponse({
    required this.content,
    required this.done,
    required this.stage,
    required this.uuid,
    required this.tps,
    required this.tokenGenerated,
  });

  @override
  int get hashCode =>
      content.hashCode ^
      done.hashCode ^
      stage.hashCode ^
      uuid.hashCode ^
      tps.hashCode ^
      tokenGenerated.hashCode;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ChatResponse &&
          runtimeType == other.runtimeType &&
          content == other.content &&
          done == other.done &&
          stage == other.stage &&
          uuid == other.uuid &&
          tps == other.tps &&
          tokenGenerated == other.tokenGenerated;
}
