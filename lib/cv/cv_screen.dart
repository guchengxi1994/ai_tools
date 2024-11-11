import 'dart:io';
import 'dart:typed_data';

import 'package:ai_tools/cv/cv_notifier.dart';
import 'package:ai_tools/cv/object_bbox.dart';
import 'package:ai_tools/src/rust/api/cv.dart';
import 'package:ai_tools/src/rust/cv/object_detect_result.dart';
import 'package:ai_tools/utils.dart';
import 'package:desktop_drop/desktop_drop.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'dart:ui' as ui;

import 'package:flutter_riverpod/flutter_riverpod.dart';

class CvScreen extends ConsumerStatefulWidget {
  const CvScreen({super.key});

  @override
  ConsumerState<CvScreen> createState() => _CvScreenState();
}

class _CvScreenState extends ConsumerState<CvScreen> {
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

  bool _isPopupVisible = false;

  void togglePopup() {
    setState(() {
      _isPopupVisible = !_isPopupVisible;
    });

    // 可选：自动隐藏弹窗
    if (_isPopupVisible) {
      Future.delayed(Duration(seconds: 3), () {
        setState(() {
          _isPopupVisible = false;
        });
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(cvStateProvider);

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
            left: 0,
            child: GestureDetector(
              onTap: () {
                togglePopup();
              },
              child: Container(
                color: Colors.white.withOpacity(0.5),
                width: MediaQuery.of(context).size.width,
                child: AnimatedContainer(
                  height: _isPopupVisible ? 500 : 100,
                  duration: Duration(milliseconds: 300),
                  child: Padding(
                    padding: EdgeInsets.only(bottom: 20, top: 20),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text("Object Detection"),
                        Wrap(
                          alignment: WrapAlignment.start,
                          children: [
                            ElevatedButton(
                              child: Text('Yolov8'),
                              onPressed: () {
                                loadYolov8(
                                        modelPath:
                                            r"D:\github_repo\ai_tools\rust\assets\yolov8n.safetensors")
                                    .then((_) {
                                  ref
                                      .read(cvStateProvider.notifier)
                                      .changeModel(CvModels.yolov8);
                                });
                              },
                            ),
                          ],
                        ),
                        Text("Classification"),
                        Wrap(
                          alignment: WrapAlignment.start,
                          children: [
                            ElevatedButton(
                              child: Text('EfficientNet'),
                              onPressed: () {
                                ref
                                    .read(cvStateProvider.notifier)
                                    .changeModel(CvModels.efficientnet);
                              },
                            ),
                            ElevatedButton(
                              child: Text('Beit'),
                              onPressed: () {
                                ref
                                    .read(cvStateProvider.notifier)
                                    .changeModel(CvModels.beit);
                              },
                            ),
                          ],
                        ),
                      ],
                    ),
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
