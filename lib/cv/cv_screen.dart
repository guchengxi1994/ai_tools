import 'dart:io';
import 'dart:typed_data';

import 'package:ai_tools/cv/classfication_box.dart';
import 'package:ai_tools/cv/cv_notifier.dart';
import 'package:ai_tools/cv/object_bbox.dart';
import 'package:ai_tools/cv/selectable_button.dart';
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
            child: state.activeModel.task == CvTask.objectDetect
                ? GestureDetector(
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
                  )
                : ClassficationBox(),
          )),
          Positioned(
            top: 0,
            left: 0,
            child: GestureDetector(
              onTap: () {
                togglePopup();
              },
              child: Container(
                color: Colors.white.withOpacity(0.9),
                width: MediaQuery.of(context).size.width - 60,
                child: AnimatedContainer(
                  height: _isPopupVisible ? 300 : 50,
                  duration: Duration(milliseconds: 300),
                  child: Padding(
                    padding: _isPopupVisible
                        ? EdgeInsets.only(bottom: 20, top: 20)
                        : EdgeInsets.zero,
                    child: _isPopupVisible
                        ? Row(
                            // mainAxisAlignment: MainAxisAlignment.start,
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Expanded(
                                  child: SingleChildScrollView(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      "Object Detection",
                                      style: TextStyle(
                                          fontWeight: FontWeight.bold),
                                    ),
                                    Wrap(
                                      alignment: WrapAlignment.start,
                                      spacing: 8,
                                      runSpacing: 8,
                                      children: [
                                        SelectableButton(
                                            model: CvModels.yolov8),
                                      ],
                                    ),
                                    SizedBox(
                                      height: 20,
                                    ),
                                    Text(
                                      "Classification",
                                      style: TextStyle(
                                          fontWeight: FontWeight.bold),
                                    ),
                                    Wrap(
                                      alignment: WrapAlignment.start,
                                      spacing: 8,
                                      runSpacing: 8,
                                      children: [
                                        SelectableButton(
                                            model: CvModels.efficientnet),
                                        SelectableButton(model: CvModels.beit),
                                      ],
                                    ),
                                  ],
                                ),
                              )),
                              SizedBox(
                                width: 50,
                                child: IconButton(
                                    onPressed: () {},
                                    icon: Container(
                                      decoration: BoxDecoration(
                                          borderRadius:
                                              BorderRadius.circular(40),
                                          color: Colors.blueAccent.shade100),
                                      width: 30,
                                      height: 30,
                                      child: Icon(Icons.play_arrow),
                                    )),
                              )
                            ],
                          )
                        : Center(
                            child: Text(
                              "click to show options",
                              style: TextStyle(fontWeight: FontWeight.bold),
                            ),
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
