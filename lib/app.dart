import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'chat_screen.dart';
import 'sidebar/sidebar.dart';
import 'sidebar/sidebar_notifier.dart';

class App extends ConsumerStatefulWidget {
  const App({super.key});

  @override
  ConsumerState<App> createState() => _AppState();
}

class _AppState extends ConsumerState<App> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          const Sidebar(),
          Expanded(
              child: PageView(
            controller: ref.read(sidebarProvider.notifier).pageController,
            children: [
              ChatScreen(),
            ],
          ))
        ],
      ),
    );
  }
}
