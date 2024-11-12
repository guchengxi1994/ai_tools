import 'package:ai_tools/sidebar/sidebar_notifier.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:window_manager/window_manager.dart';

import 'cv/cv_notifier.dart';

class AppWrapper extends ConsumerStatefulWidget {
  const AppWrapper({super.key, required this.child});
  final Widget child;

  @override
  ConsumerState<AppWrapper> createState() => _AppWrapperState();
}

class _AppWrapperState extends ConsumerState<AppWrapper> {
  @override
  Widget build(BuildContext context) {
    final state = ref.watch(sidebarProvider);
    return Scaffold(
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(30),
        child: WindowCaption(
          brightness: Theme.of(context).brightness,
          title: Row(
            children: [
              SizedBox(
                width: 25,
                height: 25,
                child: Image.asset("assets/icon.jpeg"),
              ),
              SizedBox(
                width: 10,
              ),
              Text('Ai Tools'),
              SizedBox(
                width: MediaQuery.of(context).size.width - /*perfix*/
                    100 - /*suffix*/
                    180,
              ),
              if (state == 1)
                InkWell(
                  onTap: () {
                    ref.read(cvStateProvider.notifier).changeExpand();
                  },
                  child: SizedBox(
                    width: 25,
                    height: 25,
                    child: Icon(
                      Icons.more_horiz,
                      color: Colors.black,
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
      body: widget.child,
    );
  }
}
