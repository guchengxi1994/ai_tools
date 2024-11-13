import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:ai_tools/src/rust/api/tools.dart';
import 'package:flutter/material.dart';

class ToolsScreen extends StatefulWidget {
  const ToolsScreen({super.key});

  @override
  State<ToolsScreen> createState() => _ToolsScreenState();
}

class _ToolsScreenState extends State<ToolsScreen> {
  // final stream = trainMessageStream();
  final chartStream = trainChartStream();
  // List<TrainMessage> messages = [];
  double current = 0;
  // ignore: avoid_init_to_null
  late ui.Image? image = null;

  Future<ui.Image> uint8ListToImage(Uint8List uint8list) async {
    final Completer<ui.Image> completer = Completer();

    ui.decodeImageFromList(uint8list, (ui.Image img) {
      completer.complete(img);
    });

    return completer.future;
  }

  @override
  void initState() {
    super.initState();
    chartStream.listen((event) async {
      image = await uint8ListToImage(event);

      setState(() {});
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          if (image != null)
            CustomPaint(
              painter: CustomImagePainter(image!),
            ),
          ElevatedButton(
              onPressed: () {
                trainAMlp(
                    csvPath: r"D:\github_repo\ai_tools\rust\assets\data.csv");
              },
              child: Text("Train a mlp"))
        ],
      ),
    );
  }
}

class CustomImagePainter extends CustomPainter {
  final ui.Image image;

  CustomImagePainter(this.image);

  @override
  void paint(Canvas canvas, Size size) {
    Paint paint = Paint();
    canvas.drawImage(image, Offset.zero, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) =>
      this != oldDelegate;
}
