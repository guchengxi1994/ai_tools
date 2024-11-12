import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'cv_notifier.dart';

class SelectableButton extends ConsumerWidget {
  const SelectableButton({super.key, required this.model});
  final CvModels model;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(cvStateProvider);

    return InkWell(
      borderRadius: BorderRadius.circular(8),
      onTap: () {
        model.onSelect(ref);
      },
      child: Container(
        padding: EdgeInsets.all(5),
        decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(8),
            color: state.activeModel == model ? Colors.blue : Colors.grey[200]),
        child: FittedBox(
          child: Row(
            children: [
              Text(model.name,
                  style: TextStyle(
                      color: state.activeModel == model
                          ? Colors.white
                          : Colors.black)),
              SizedBox(width: 5),
              state.activeModel == model
                  ? Icon(
                      Icons.check,
                      color: Colors.white,
                      size: 20,
                    )
                  : state.loadingModel == model
                      ? SizedBox(
                          width: 15,
                          height: 15,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.blue,
                          ),
                        )
                      : Container()
            ],
          ),
        ),
      ),
    );
  }
}
