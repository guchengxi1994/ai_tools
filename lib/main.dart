import 'package:ai_tools/src/rust/api/llm.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:ai_tools/src/rust/frb_generated.dart';
import 'package:file_selector/file_selector.dart';

import 'src/rust/api/cv.dart';

Future<void> main() async {
  await RustLib.init();
  runApp(const MyApp());
}

XTypeGroup typeGroup = const XTypeGroup(
  label: 'images',
);

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('flutter_rust_bridge quickstart')),
        body: Column(
          children: [
            ElevatedButton(
                onPressed: () async {
                  final file = await openFile(acceptedTypeGroups: [typeGroup]);
                  if (file != null) {
                    if (kDebugMode) {
                      print("file path: ${file.path}");
                      print(await classifyImage(
                          s: file.path,
                          modelPath:
                              r"D:\github_repo\ai_tools\rust\assets\efficientnet-lite4-s.onnx"));
                    }
                  }
                },
                child: Text("Select and infer")),
            ElevatedButton(
                onPressed: () async {
                  qwen2Chat(
                      userPrompt: "请解释机器学习的基本概念。",
                      systemPrompt: "请用简洁、专业的语言回答问题。",
                      modelPath:
                          r"D:\github_repo\ai_tools\rust\assets\Qwen2-0___5B-Instruct");
                },
                child: Text("qwen2 chat test"))
          ],
        ),
      ),
    );
  }
}
