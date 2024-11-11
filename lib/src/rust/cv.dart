// This file is automatically generated, so please do not edit it.
// @generated by `flutter_rust_bridge`@ 2.5.1.

// ignore_for_file: invalid_use_of_internal_member, unused_import, unnecessary_import

import 'frb_generated.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';

class ClassificationResults {
  final List<(String, double)> results;
  final double duration;

  const ClassificationResults({
    required this.results,
    required this.duration,
  });

  @override
  int get hashCode => results.hashCode ^ duration.hashCode;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ClassificationResults &&
          runtimeType == other.runtimeType &&
          results == other.results &&
          duration == other.duration;
}
