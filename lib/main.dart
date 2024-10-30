import 'package:ai_tools/app.dart';
import 'package:flutter/material.dart';
import 'package:ai_tools/src/rust/frb_generated.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:toastification/toastification.dart';

Future<void> main() async {
  await RustLib.init();
  runApp(const ProviderScope(
    child: ToastificationWrapper(
      child: MyApp(),
    ),
  ));
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: App(),
    );
  }
}
