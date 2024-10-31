import 'package:ai_tools/messagebox/chat_response.dart';
import 'package:ai_tools/src/rust/api/llm.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'input/input_field.dart';
import 'messagebox/llm_request_messagebox.dart';
import 'messagebox/message_notifier.dart';
import 'messagebox/messagebox_state.dart';

class ChatScreen extends ConsumerStatefulWidget {
  const ChatScreen({super.key});

  @override
  ConsumerState<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends ConsumerState<ChatScreen> {
  final responseStream = chatResponseStream();

  @override
  void initState() {
    super.initState();
    responseStream.listen((event) {
      ref
          .read(messageProvider.notifier)
          .updateMessageBox(ChatResponse.fromRustChatResponse(event));
    });
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(messageProvider);

    return Column(
      children: [
        Flexible(
            child: SizedBox.expand(
          child: SingleChildScrollView(
            controller: ref.read(messageProvider.notifier).scrollController,
            padding: const EdgeInsets.only(left: 20, right: 20),
            child: Column(
              children: state.messageBox.map((e) => e.toWidget()).toList(),
            ),
          ),
        )),
        InputField(onSubmit: (s) => _handleInputMessage(s, state))
      ],
    );
  }

  _handleInputMessage(String s, MessageState state) async {
    if (state.isLoading) {
      return;
    }

    ref
        .read(messageProvider.notifier)
        .addMessageBox(RequestMessageBox(content: s));

    qwen2Chat(
        userPrompt: s,
        modelPath:
            r"D:\github_repo\ai_tools\rust\assets\Qwen2___5-0___5B-Instruct");

    ref.read(messageProvider.notifier).jumpToMax();
  }
}
