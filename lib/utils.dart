import 'package:flutter/material.dart';
import 'package:toastification/toastification.dart';

// ignore: depend_on_referenced_packages
import 'package:logging/logging.dart';

final logger = Logger("at");

class ToastUtils {
  ToastUtils._();

  static void info(BuildContext? context,
      {required String title, String? descryption, VoidCallback? onTap}) {
    toastification.show(
      context: context,
      type: ToastificationType.success,
      style: ToastificationStyle.flatColored,
      autoCloseDuration: const Duration(seconds: 2),
      title: Text(title),
      // you can also use RichText widget for title and description parameters
      description: descryption != null
          ? RichText(text: TextSpan(text: descryption))
          : null,
      alignment: Alignment.topRight,
      direction: TextDirection.ltr,
      animationDuration: const Duration(milliseconds: 300),
      animationBuilder: (context, animation, alignment, child) {
        return FadeTransition(opacity: animation, child: child);
      },
      icon: const Icon(
        Icons.info,
        color: Colors.blueAccent,
      ),
      primaryColor: Colors.blueAccent,
      backgroundColor: Colors.white,
      foregroundColor: Colors.black,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 16),
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      // borderRadius: BorderRadius.circular(12),
      // boxShadow: const [
      //   BoxShadow(
      //     color: Color(0x07000000),
      //     blurRadius: 16,
      //     offset: Offset(0, 16),
      //     spreadRadius: 0,
      //   )
      // ],
      showProgressBar: true,
      closeButtonShowType: CloseButtonShowType.onHover,
      closeOnClick: false,
      pauseOnHover: true,
      dragToClose: true,
      // applyBlurEffect: true,
      callbacks: ToastificationCallbacks(
        onTap: (toastItem) => onTap,
        onCloseButtonTap: (toastItem) {
          toastification.dismiss(toastItem);
        },
      ),
    );
  }

  static void sucess(BuildContext? context,
      {required String title, String? descryption, VoidCallback? onTap}) {
    toastification.show(
      context: context,
      type: ToastificationType.success,
      style: ToastificationStyle.flatColored,
      autoCloseDuration: const Duration(seconds: 2),
      title: Text(title),
      // you can also use RichText widget for title and description parameters
      description: descryption != null
          ? RichText(text: TextSpan(text: descryption))
          : null,
      alignment: Alignment.topRight,
      direction: TextDirection.ltr,
      animationDuration: const Duration(milliseconds: 300),
      animationBuilder: (context, animation, alignment, child) {
        return FadeTransition(opacity: animation, child: child);
      },
      icon: const Icon(
        Icons.check,
        color: Colors.green,
      ),
      primaryColor: Colors.green,
      backgroundColor: Colors.white,
      foregroundColor: Colors.black,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 16),
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      // borderRadius: BorderRadius.circular(12),
      // boxShadow: const [
      //   BoxShadow(
      //     color: Color(0x07000000),
      //     blurRadius: 16,
      //     offset: Offset(0, 16),
      //     spreadRadius: 0,
      //   )
      // ],
      showProgressBar: true,
      closeButtonShowType: CloseButtonShowType.onHover,
      closeOnClick: false,
      pauseOnHover: true,
      dragToClose: true,
      // applyBlurEffect: true,
      callbacks: ToastificationCallbacks(
        onTap: (toastItem) => onTap,
        onCloseButtonTap: (toastItem) {
          toastification.dismiss(toastItem);
        },
      ),
    );
  }

  static void error(BuildContext? context,
      {required String title, String? descryption, VoidCallback? onTap}) {
    toastification.show(
      context: context,
      type: ToastificationType.error,
      style: ToastificationStyle.flatColored,
      autoCloseDuration: const Duration(seconds: 5),
      title: Text(title),
      // you can also use RichText widget for title and description parameters
      description: descryption != null
          ? RichText(text: TextSpan(text: descryption))
          : null,
      alignment: Alignment.topRight,
      direction: TextDirection.ltr,
      animationDuration: const Duration(milliseconds: 300),
      animationBuilder: (context, animation, alignment, child) {
        return FadeTransition(opacity: animation, child: child);
      },
      icon: const Icon(
        Icons.error,
        color: Colors.red,
      ),
      primaryColor: Colors.red,
      backgroundColor: Colors.white,
      foregroundColor: Colors.black,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 16),
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      // borderRadius: BorderRadius.circular(12),
      // boxShadow: const [
      //   BoxShadow(
      //     color: Color(0x07000000),
      //     blurRadius: 16,
      //     offset: Offset(0, 16),
      //     spreadRadius: 0,
      //   )
      // ],
      showProgressBar: true,
      closeButtonShowType: CloseButtonShowType.onHover,
      closeOnClick: false,
      pauseOnHover: true,
      dragToClose: true,
      // applyBlurEffect: true,
      callbacks: ToastificationCallbacks(
        onTap: (toastItem) => onTap,
        onCloseButtonTap: (toastItem) {
          toastification.dismiss(toastItem);
        },
      ),
    );
  }
}

class HexColor extends Color {
  static int _getColorFromHex(String hexColor) {
    hexColor = hexColor.toUpperCase().replaceAll("#", "");
    if (hexColor.length == 6) {
      hexColor = "FF$hexColor";
    }
    return int.parse(hexColor, radix: 16);
  }

  HexColor(final String hexColor) : super(_getColorFromHex(hexColor));
}

class Styles {
  Styles._();

  static Color black515A6E = HexColor('#515A6E');
  static Color divider = HexColor('#DCDFE6');
  static Color blue247AF2 = HexColor('#247AF2');
}
