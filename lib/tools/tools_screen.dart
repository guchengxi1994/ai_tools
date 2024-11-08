import 'package:ai_tools/src/rust/api/tools.dart';
import 'package:ai_tools/src/rust/tools.dart';
import 'package:ai_tools/utils.dart';
import 'package:flutter/material.dart';
import 'package:flutter_animation_progress_bar/flutter_animation_progress_bar.dart';
import 'package:syncfusion_flutter_charts/charts.dart';

class ToolsScreen extends StatefulWidget {
  const ToolsScreen({super.key});

  @override
  State<ToolsScreen> createState() => _ToolsScreenState();
}

class _ToolsScreenState extends State<ToolsScreen> {
  final stream = trainMessageStream();
  List<TrainMessage> messages = [];
  double current = 0;

  @override
  void initState() {
    super.initState();
    stream.listen((event) {
      logger.info("loss ${event.loss}");
      setState(() {
        messages.add(event);
        current = event.epoch.toInt() / 10000 * 100;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          if (messages.isNotEmpty)
            SfCartesianChart(
              primaryXAxis: NumericAxis(
                name: "epoch",
                title: AxisTitle(text: "epoch"),
              ),
              primaryYAxis: NumericAxis(
                name: "loss",
                title: AxisTitle(text: "loss"),
              ),
              series: <CartesianSeries>[
                LineSeries<TrainMessage, double>(
                    dataSource: messages,
                    xValueMapper: (TrainMessage e, _) => e.epoch.toInt() * 1.0,
                    yValueMapper: (TrainMessage e, _) => e.loss)
              ],
            ),
          if (messages.isNotEmpty)
            FAProgressBar(
              direction: Axis.horizontal,
              displayText: "%",
              currentValue: current,
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
