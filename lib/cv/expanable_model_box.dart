import 'package:ai_tools/cv/cv_notifier.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'selectable_button.dart';

class ExpanableModelBox extends ConsumerStatefulWidget {
  const ExpanableModelBox({super.key});

  @override
  ConsumerState<ExpanableModelBox> createState() => _ExpanableModelBoxState();
}

class _ExpanableModelBoxState extends ConsumerState<ExpanableModelBox> {
  @override
  Widget build(BuildContext context) {
    final state = ref.watch(cvStateProvider.select((v) => v.isExpanded));

    return Container(
      color: Colors.white.withOpacity(0.9),
      width: MediaQuery.of(context).size.width - 60,
      child: AnimatedContainer(
        height: state ? 300 : 0,
        duration: Duration(milliseconds: 300),
        child: Padding(
          padding:
              state ? EdgeInsets.only(bottom: 20, top: 20) : EdgeInsets.zero,
          child: state
              ? Row(
                  // mainAxisAlignment: MainAxisAlignment.start,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                        child: SingleChildScrollView(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            "Object Detection",
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          Wrap(
                            alignment: WrapAlignment.start,
                            spacing: 8,
                            runSpacing: 8,
                            children: [
                              SelectableButton(model: CvModels.yolov8),
                            ],
                          ),
                          SizedBox(
                            height: 20,
                          ),
                          Text(
                            "Classification",
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          Wrap(
                            alignment: WrapAlignment.start,
                            spacing: 8,
                            runSpacing: 8,
                            children: [
                              SelectableButton(model: CvModels.efficientnet),
                              SelectableButton(model: CvModels.beit),
                            ],
                          ),
                        ],
                      ),
                    )),
                    SizedBox(
                      width: 50,
                      child: IconButton(
                          onPressed: () {},
                          icon: Container(
                            decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(40),
                                color: Colors.blueAccent.shade100),
                            width: 30,
                            height: 30,
                            child: Icon(Icons.play_arrow),
                          )),
                    )
                  ],
                )
              : SizedBox(),
        ),
      ),
    );
  }
}
