import 'dart:io';
import 'dart:typed_data';

import 'package:ai_tools/cv/object_bbox.dart';
import 'package:ai_tools/src/rust/api/cv.dart';
import 'package:ai_tools/src/rust/cv/object_detect_result.dart';
import 'package:ai_tools/utils.dart';
import 'package:desktop_drop/desktop_drop.dart';
import 'package:expand_widget/expand_widget.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'dart:ui' as ui;

class CvScreen extends StatefulWidget {
  const CvScreen({super.key});

  @override
  State<CvScreen> createState() => _CvScreenState();
}

class _CvScreenState extends State<CvScreen> {
  final GlobalKey _key = GlobalKey();

  final loadModelStream = loadModelStateStream();

  @override
  void initState() {
    super.initState();
    loadModelStream.listen((event) {
      ToastUtils.info(null, title: event.replaceAll("=>", " "));
    });
  }

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
        yolov8Detect(img: pngBytes).then((v) {
          setState(() {
            results = v;
          });
        });
      }
    } catch (e) {
      logger.severe("Error: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
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
          Positioned(
            top: 0,
            child: Container(
              color: Colors.white.withOpacity(0.5),
              width: MediaQuery.of(context).size.width,
              child: ExpandChild(
                collapsedVisibilityFactor: 0.5,
                child: Padding(
                  padding: EdgeInsets.only(bottom: 20, top: 20),
                  child: Wrap(
                    alignment: WrapAlignment.start,
                    children: [
                      ElevatedButton(
                        child: Text('Yolov8'),
                        onPressed: () {
                          yolov8Init();
                        },
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
          ...results.map((v) => Positioned(
              top: v.ymin * 1.0,
              left: v.xmin * 1.0,
              child: ObjectBbox(bbox: v)))
        ],
      ),
    );
  }
}
