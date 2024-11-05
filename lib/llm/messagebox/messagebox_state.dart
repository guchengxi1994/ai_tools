import 'messagebox.dart';

enum ServerState { running, loading, none }

class MessageState {
  List<MessageBox> messageBox;
  bool isLoading = false;
  bool useHistory;
  bool useThoughtChain;
  ServerState serverState;

  MessageState({
    this.messageBox = const [],
    this.isLoading = false,
    this.useHistory = false,
    this.useThoughtChain = false,
    this.serverState = ServerState.none,
  });

  MessageState copyWith({
    List<MessageBox>? messageBox,
    bool? isLoading,
    bool? useHistory,
    bool? useThoughtChain,
    ServerState? serverState,
  }) {
    return MessageState(
      messageBox: messageBox ?? this.messageBox,
      isLoading: isLoading ?? this.isLoading,
      useHistory: useHistory ?? this.useHistory,
      useThoughtChain: useThoughtChain ?? this.useThoughtChain,
      serverState: serverState ?? this.serverState,
    );
  }
}
