import 'package:ai_tools/app.dart';
import 'package:ai_tools/isar/database.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:ai_tools/src/rust/frb_generated.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:toastification/toastification.dart';
// ignore: depend_on_referenced_packages
import 'package:logging/logging.dart';
import 'package:window_manager/window_manager.dart';

import 'app_wrapper.dart';
import 'src/rust/api/simple.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // 必须加上这一行。
  await windowManager.ensureInitialized();
  await RustLib.init();
  await initLogger();
  Logger.root.level = Level.ALL; // defaults to Level.INFO
  Logger.root.onRecord.listen((record) {
    if (kDebugMode) {
      print('${record.level.name}: ${record.time}: ${record.message}');
    }
  });
  IsarDatabase database = IsarDatabase();
  await database.initialDatabase();

  WindowOptions windowOptions = WindowOptions(
    center: false,
    backgroundColor: Colors.transparent,
    skipTaskbar: false,
    titleBarStyle: TitleBarStyle.hidden,
  );
  windowManager.waitUntilReadyToShow(windowOptions, () async {
    await windowManager.show();
    await windowManager.focus();
  });

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
      home: AppWrapper(
        child: App(),
      ),
    );
  }
}
