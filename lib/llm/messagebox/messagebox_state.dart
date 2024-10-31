import 'messagebox.dart';

class MessageState {
  List<MessageBox> messageBox;
  bool isLoading = false;
  bool useHistory;
  bool useThoughtChain;

  MessageState({
    this.messageBox = const [],
    this.isLoading = false,
    this.useHistory = false,
    this.useThoughtChain = false,
  });

  MessageState copyWith({
    List<MessageBox>? messageBox,
    bool? isLoading,
    bool? useHistory,
    bool? useThoughtChain,
  }) {
    return MessageState(
      messageBox: messageBox ?? this.messageBox,
      isLoading: isLoading ?? this.isLoading,
      useHistory: useHistory ?? this.useHistory,
      useThoughtChain: useThoughtChain ?? this.useThoughtChain,
    );
  }
}
