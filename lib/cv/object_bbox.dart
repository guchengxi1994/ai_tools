import 'package:ai_tools/src/rust/cv/object_detect_result.dart';
import 'package:flutter/material.dart';

class ObjectBbox extends StatefulWidget {
  const ObjectBbox({super.key, required this.bbox});
  final ObjectDetectResult bbox;

  @override
  State<ObjectBbox> createState() => _ObjectBboxState();
}

class _ObjectBboxState extends State<ObjectBbox> {
  bool hovering = false;

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (event) => setState(() => hovering = true),
      onExit: (event) => setState(() => hovering = false),
      child: Container(
        width: widget.bbox.width * 1.0,
        height: widget.bbox.height * 1.0,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(4),
          border: Border.all(
              color: hovering ? Colors.blue : Colors.transparent, width: 2),
          color: hovering
              ? Colors.blue.withOpacity(0.5)
              : Colors.blue.withOpacity(0.3),
        ),
        child: hovering
            ? Center(
                child: Text(
                  widget.bbox.className,
                  style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 20),
                ),
              )
            : null,
      ),
    );
  }
}
