// This file is automatically generated, so please do not edit it.
// @generated by `flutter_rust_bridge`@ 2.5.1.

// ignore_for_file: invalid_use_of_internal_member, unused_import, unnecessary_import

import '../frb_generated.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';

Future<String> classifyImage(
        {required String s,
        required String modelPath,
        List<String>? classes}) =>
    RustLib.instance.api
        .crateApiCvClassifyImage(s: s, modelPath: modelPath, classes: classes);
