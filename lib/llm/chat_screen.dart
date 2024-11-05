import 'package:ai_tools/llm/messagebox/chat_response.dart';
import 'package:ai_tools/src/rust/api/llm.dart';
import 'package:ai_tools/utils.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../input/input_field.dart';
import 'crazy_switch.dart';
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
  final stateStream = serverStateStream();

  @override
  void initState() {
    super.initState();
    responseStream.listen((event) {
      ref
          .read(messageProvider.notifier)
          .updateMessageBox(ChatResponse.fromRustChatResponse(event));
    });
    stateStream.listen((event) {
      if (event == "init") {
        ToastUtils.info(null, title: "model loading");
        ref
            .read(messageProvider.notifier)
            .changeServerState(ServerState.loading);
      } else if (event == "model loaded") {
        ToastUtils.info(null, title: "server ready");
        ref
            .read(messageProvider.notifier)
            .changeServerState(ServerState.running);
      } else {
        ToastUtils.info(null, title: "server stopped");
        ref.read(messageProvider.notifier).changeServerState(ServerState.none);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(messageProvider);

    return Column(
      children: [
        SizedBox(
          width: MediaQuery.of(context).size.width,
          height: 50,
          child: Center(
            child: Row(
              mainAxisAlignment: MainAxisAlignment.start,
              children: [
                Spacer(),
                Container(
                  padding:
                      EdgeInsets.only(left: 30, right: 30, top: 5, bottom: 5),
                  decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: Colors.grey[300]!)),
                  child: Text(
                    'Qwen2.5 0.5B',
                    style: TextStyle(
                        color: Colors.black, fontWeight: FontWeight.bold),
                  ),
                ),
                SizedBox(
                  width: 20,
                ),
                Text(
                  "Use thought chain",
                  style: TextStyle(
                      color: Colors.black, fontWeight: FontWeight.bold),
                ),
                SizedBox(
                  width: 10,
                ),
                CrazySwitch(
                  current: state.useThoughtChain,
                  onChanged: (value) {
                    ref
                        .read(messageProvider.notifier)
                        .updateUseThoughtChain(value);
                  },
                ),
                SizedBox(
                  width: 20,
                ),
                Text(
                  "With history",
                  style: TextStyle(
                      color: Colors.black, fontWeight: FontWeight.bold),
                ),
                SizedBox(
                  width: 10,
                ),
                CrazySwitch(
                  current: state.useHistory,
                  onChanged: (value) {
                    ref.read(messageProvider.notifier).updateWithHistory(value);
                  },
                ),
                Spacer(),
                IconButton(
                    onPressed: () {
                      if (state.serverState == ServerState.running) {
                        ref.read(messageProvider.notifier).stopserver();
                      } else if (state.serverState == ServerState.none) {
                        ref.read(messageProvider.notifier).runserver();
                      }
                    },
                    icon: Container(
                      decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(40),
                          color: Colors.blueAccent.shade100),
                      width: 30,
                      height: 30,
                      child: state.serverState == ServerState.running
                          ? Icon(Icons.stop)
                          : state.serverState == ServerState.loading
                              ? Icon(Icons.hourglass_empty)
                              : Icon(Icons.play_arrow),
                    ))
              ],
            ),
          ),
        ),
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

    ref.read(messageProvider.notifier).chat(s).then((_) {
      ref.read(messageProvider.notifier).jumpToMax();
    });
  }
}
