import 'messagebox.dart';

enum ServerState { running, loading, none }

enum LLMState { loading, loaded, none }

class MessageState {
  List<MessageBox> messageBox;
  bool isLoading = false;
  bool useHistory;
  bool useThoughtChain;
  ServerState serverState;
  LLMState llmState;

  MessageState({
    this.messageBox = const [],
    this.isLoading = false,
    this.useHistory = false,
    this.useThoughtChain = false,
    this.serverState = ServerState.none,
    this.llmState = LLMState.none,
  });

  MessageState copyWith({
    List<MessageBox>? messageBox,
    bool? isLoading,
    bool? useHistory,
    bool? useThoughtChain,
    ServerState? serverState,
    LLMState? llmState,
  }) {
    return MessageState(
      messageBox: messageBox ?? this.messageBox,
      isLoading: isLoading ?? this.isLoading,
      useHistory: useHistory ?? this.useHistory,
      useThoughtChain: useThoughtChain ?? this.useThoughtChain,
      serverState: serverState ?? this.serverState,
      llmState: llmState ?? this.llmState,
    );
  }
}
