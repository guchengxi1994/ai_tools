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
