// This file is automatically generated, so please do not edit it.
// @generated by `flutter_rust_bridge`@ 2.5.1.

// ignore_for_file: invalid_use_of_internal_member, unused_import, unnecessary_import

import '../frb_generated.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';

class ObjectDetectResult {
  final BigInt classId;
  final String className;
  final double confidence;
  final int xmin;
  final int ymin;
  final int width;
  final int height;

  const ObjectDetectResult({
    required this.classId,
    required this.className,
    required this.confidence,
    required this.xmin,
    required this.ymin,
    required this.width,
    required this.height,
  });

  @override
  int get hashCode =>
      classId.hashCode ^
      className.hashCode ^
      confidence.hashCode ^
      xmin.hashCode ^
      ymin.hashCode ^
      width.hashCode ^
      height.hashCode;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ObjectDetectResult &&
          runtimeType == other.runtimeType &&
          classId == other.classId &&
          className == other.className &&
          confidence == other.confidence &&
          xmin == other.xmin &&
          ymin == other.ymin &&
          width == other.width &&
          height == other.height;
}