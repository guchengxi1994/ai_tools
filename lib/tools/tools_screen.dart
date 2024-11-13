import 'dart:typed_data';

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
  Uint8List? chartData = null;

  @override
  void initState() {
    super.initState();
    chartStream.listen((event) {
      setState(() {
        chartData = event;
      });
      // logger.info("loss ${event.loss}");
      // setState(() {
      //   messages.add(event);
      //   current = event.epoch.toInt() / 10000 * 100;
      // });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          if (chartData != null) Image.memory(chartData!),
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
