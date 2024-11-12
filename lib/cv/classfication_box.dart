import 'dart:io';

import 'package:ai_tools/src/rust/api/cv.dart';
import 'package:ai_tools/src/rust/cv.dart';
import 'package:ai_tools/utils.dart';
import 'package:desktop_drop/desktop_drop.dart';
import 'package:flutter/material.dart';
// ignore: depend_on_referenced_packages
import 'package:collection/collection.dart';

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
                    .mapIndexed((i, v) => Container(
                          alignment: Alignment.topLeft,
                          margin: EdgeInsets.only(bottom: 5),
                          padding: EdgeInsets.all(5),
                          child: RichText(
                              text: TextSpan(children: [
                            TextSpan(
                                text: "${i + 1}. ",
                                style: TextStyle(
                                    color: Colors.green,
                                    fontWeight: FontWeight.bold)),
                            TextSpan(
                                text: "classname: ",
                                style: TextStyle(
                                    color: Colors.black,
                                    fontWeight: FontWeight.bold)),
                            TextSpan(
                                text: "${v.$1} ",
                                style: TextStyle(color: Colors.blueAccent)),
                            TextSpan(
                                text: ",confidence: ",
                                style: TextStyle(
                                    color: Colors.black,
                                    fontWeight: FontWeight.bold)),
                            TextSpan(
                                text: "${(v.$2 * 100).toStringAsFixed(2)}%",
                                style: TextStyle(color: Colors.blueAccent))
                          ])),
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
