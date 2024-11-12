import 'package:ai_tools/cv/classfication_box.dart';
import 'package:ai_tools/cv/cv_notifier.dart';
import 'package:ai_tools/cv/expanable_model_box.dart';
import 'package:ai_tools/cv/object_detect_box.dart';
import 'package:ai_tools/src/rust/api/cv.dart';
import 'package:ai_tools/utils.dart';
import 'package:flutter/material.dart';

import 'package:flutter_riverpod/flutter_riverpod.dart';

class CvScreen extends ConsumerStatefulWidget {
  const CvScreen({super.key});

  @override
  ConsumerState<CvScreen> createState() => _CvScreenState();
}

class _CvScreenState extends ConsumerState<CvScreen> {
  final loadModelStream = loadModelStateStream();

  @override
  void initState() {
    super.initState();
    loadModelStream.listen((event) {
      ToastUtils.info(null, title: event.replaceAll("=>", " "));
    });
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(cvStateProvider);

    return Scaffold(
      body: GestureDetector(
        onTap: () {
          if (state.isExpanded) {
            ref.read(cvStateProvider.notifier).changeExpand();
          }
        },
        child: Stack(
          children: [
            SizedBox.expand(
                child: Padding(
              padding: EdgeInsets.all(20),
              child: state.activeModel.task == CvTask.objectDetect
                  ? ObjectDetectBox()
                  : ClassficationBox(),
            )),
            Positioned(
              top: 0,
              left: 0,
              child: ExpanableModelBox(),
            ),
          ],
        ),
      ),
    );
  }
}
