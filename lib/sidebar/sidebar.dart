import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'sidebar_item.dart';
import 'sidebar_notifier.dart';

class Sidebar extends ConsumerWidget {
  const Sidebar({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    late final List<SidebarItem> items = [
      SidebarItem(
          icon: const Icon(
            Icons.chat,
            color: Colors.blueAccent,
          ),
          iconInactive: const Icon(Icons.chat),
          onClick: (v) {
            ref.read(sidebarProvider.notifier).setIndex(v);
          },
          index: 0,
          title: "LLM"),
      SidebarItem(
          icon: const Icon(
            Icons.visibility,
            color: Colors.blueAccent,
          ),
          iconInactive: const Icon(Icons.visibility),
          onClick: (v) {
            ref.read(sidebarProvider.notifier).setIndex(v);
          },
          index: 1,
          title: "CV"),
      SidebarItem(
          icon: const Icon(
            Icons.functions,
            color: Colors.blueAccent,
          ),
          iconInactive: const Icon(Icons.functions),
          onClick: (v) {
            ref.read(sidebarProvider.notifier).setIndex(v);
          },
          index: 2,
          title: "Tools"),
    ];

    return Padding(
      padding: const EdgeInsets.all(10),
      child: Material(
        elevation: 10,
        borderRadius: BorderRadius.circular(10),
        child: Container(
          width: 40,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(10),
          ),
          child: Column(
            children: items.map((v) => SidebarItemWidget(item: v)).toList(),
          ),
        ),
      ),
    );
  }
}
