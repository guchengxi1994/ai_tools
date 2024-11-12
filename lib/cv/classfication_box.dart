import 'dart:io';

import 'package:ai_tools/src/rust/api/cv.dart';
import 'package:ai_tools/src/rust/cv.dart';
import 'package:ai_tools/utils.dart';
import 'package:desktop_drop/desktop_drop.dart';
import 'package:flutter/material.dart';

class ClassficationBox extends StatefulWidget {
  const ClassficationBox({super.key});

  @override
  State<ClassficationBox> createState() => _ClassficationBoxState();
}

class _ClassficationBoxState extends State<ClassficationBox> {
  late String filepath = "";
  // ignore: avoid_init_to_null
  late ClassificationResults? results = null;

  Future process() async {
    if (filepath == "") return;

    runClassification(img: filepath).then((v) {
      setState(() {
        results = v;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.all(10),
      child: Row(
        children: [
          Expanded(
            flex: 3,
            child: GestureDetector(
              onTap: filepath == ""
                  ? null
                  : () {
                      process();
                    },
              child: MouseRegion(
                cursor: filepath == ""
                    ? SystemMouseCursors.forbidden
                    : SystemMouseCursors.click,
                child: DropTarget(
                    onDragDone: (details) {
                      if (details.files.isNotEmpty) {
                        logger.info(details.files.first.path);
                        setState(() {
                          results = null;
                          filepath = details.files.first.path;
                        });
                      }
                    },
                    child: Container(
                      margin: EdgeInsets.only(right: 20),
                      decoration: BoxDecoration(
                          image: filepath == ""
                              ? null
                              : DecorationImage(
                                  image: FileImage(File(filepath))),
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all()),
                    )),
              ),
            ),
          ),
          Expanded(
            flex: 1,
            child: Container(
              decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all()),
              child: Column(
                children: (results?.results ?? [])
                    .map((v) => Row(
                          children: [
                            Text(v.$1),
                            Text(" ${(v.$2 * 100).toStringAsFixed(2)}%")
                          ],
                        ))
                    .toList(),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
