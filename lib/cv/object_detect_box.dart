import 'dart:io';

import 'package:ai_tools/src/rust/api/cv.dart';
import 'package:ai_tools/src/rust/cv/object_detect_result.dart';
import 'package:ai_tools/utils.dart';
import 'package:desktop_drop/desktop_drop.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'dart:ui' as ui;
import 'dart:typed_data';

import 'object_bbox.dart';

class ObjectDetectBox extends StatefulWidget {
  const ObjectDetectBox({super.key});

  @override
  State<ObjectDetectBox> createState() => _ObjectDetectBoxState();
}

class _ObjectDetectBoxState extends State<ObjectDetectBox> {
  final GlobalKey _key = GlobalKey();

  late String filepath = "";
  late List<ObjectDetectResult> results = [];

  Future<void> _captureAndProcessPng() async {
    try {
      // 获取 RenderRepaintBoundary
      RenderRepaintBoundary boundary =
          _key.currentContext?.findRenderObject() as RenderRepaintBoundary;

      // 转换为图片
      ui.Image image = await boundary.toImage(pixelRatio: 1.0);
      ByteData? byteData =
          await image.toByteData(format: ui.ImageByteFormat.png);
      if (byteData != null) {
        // 获取 PNG 字节数据
        final pngBytes = byteData.buffer.asUint8List();
        runDetectInBytes(img: pngBytes).then((v) {
          setState(() {
            results = v.results;
          });
        });
      }
    } catch (e) {
      logger.severe("Error: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        SizedBox.expand(
            child: Padding(
          padding: EdgeInsets.all(20),
          child: GestureDetector(
            onTap: filepath == ""
                ? null
                : () {
                    results.clear();
                    _captureAndProcessPng();
                  },
            child: MouseRegion(
              cursor: filepath == ""
                  ? SystemMouseCursors.forbidden
                  : SystemMouseCursors.click,
              child: DropTarget(
                  onDragDone: (details) {
                    if (details.files.isNotEmpty) {
                      logger.info(details.files.first.path);
                      setState(() {
                        results.clear();
                        filepath = details.files.first.path;
                      });
                    }
                  },
                  child: RepaintBoundary(
                    key: _key,
                    child: Container(
                      decoration: BoxDecoration(
                          image: filepath == ""
                              ? null
                              : DecorationImage(
                                  image: FileImage(File(filepath))),
                          borderRadius: BorderRadius.circular(10),
                          color: Colors.grey[200]),
                    ),
                  )),
            ),
          ),
        )),
        ...results.map((v) => Positioned(
            top: v.ymin * 1.0, left: v.xmin * 1.0, child: ObjectBbox(bbox: v)))
      ],
    );
  }
}
