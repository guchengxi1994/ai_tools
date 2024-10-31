import 'messagebox.dart';

class MessageState {
  List<MessageBox> messageBox;
  bool isLoading = false;

  MessageState({
    this.messageBox = const [],
    this.isLoading = false,
  });
}
