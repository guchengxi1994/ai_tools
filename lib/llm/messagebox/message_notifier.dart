import 'package:ai_tools/isar/chat_history.dart';
import 'package:ai_tools/isar/database.dart';
import 'package:ai_tools/src/rust/api/llm.dart';
import 'package:ai_tools/utils.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:ai_tools/src/rust/llm.dart' as llm;
import 'package:isar/isar.dart';

import 'chat_response.dart';
import 'llm_response_messagebox.dart';
import 'messagebox.dart';
import 'messagebox_state.dart';

class MessageNotifier extends AutoDisposeNotifier<MessageState> {
  final ScrollController scrollController = ScrollController();
  final IsarDatabase database = IsarDatabase();

  changeServerState(ServerState s) {
    if (s == state.serverState) {
      return;
    }

    state = state.copyWith(serverState: s);
  }

  void runserver() {
    if (state.serverState == ServerState.running ||
        state.serverState == ServerState.loading) {
      return;
    }

    runServer(
        modelPath:
            r"D:\github_repo\ai_tools\rust\assets\Qwen2___5-0___5B-Instruct");
  }

  void stopserver() {
    if (state.serverState != ServerState.running) {
      return;
    }

    stopServer();
  }

  @override
  MessageState build() {
    return MessageState();
  }

  addMessageBox(MessageBox box) {
    if (state.isLoading) {
      return;
    }

    final l = List<MessageBox>.from(state.messageBox)..add(box);

    // state = MessageState(
    //   messageBox: l,
    //   isLoading: state.isLoading,
    // );
    state = state.copyWith(
      isLoading: state.isLoading,
      messageBox: l,
    );

    scrollController.jumpTo(
      scrollController.position.maxScrollExtent,
    );
  }

  jumpToMax() {
    scrollController.jumpTo(
      scrollController.position.maxScrollExtent,
    );
  }

  Future chat(
    String q,
  ) async {
    await saveHistory(q, ChatRole.user);
    List<ChatHistory> list;
    if (state.useHistory) {
      list = ((await database.isar!.chatHistorys
                  .where()
                  .sortByCreateAtDesc()
                  .limit(6)
                  .findAll())
              .reversed)
          .toList();
    } else {
      list = [
        ChatHistory()
          ..content = q
          ..role = ChatRole.user
      ];
    }

    final String prompt = formatPromptWithHistory(list);
    logger.info("prompt: $prompt");
    qwen2PromptChat(prompt: prompt);
  }

  saveHistory(String content, ChatRole role) async {
    ChatHistory history = ChatHistory()
      ..content = content
      ..role = role;
    await database.isar!.writeTxn(() async {
      await database.isar!.chatHistorys.put(history);
    });
  }

  String formatPromptWithHistory(List<ChatHistory> history) {
    List<llm.ChatMessage> messages = [];
    for (var item in history) {
      messages.add(llm.ChatMessage(
        role: item.role == ChatRole.user ? "user" : "assistant",
        content: item.content,
      ));
    }
    if (state.useThoughtChain) {
      return formatPromptWithThoughtChain(
          messages: llm.ChatMessages(field0: messages));
    }
    return formatPrompt(messages: llm.ChatMessages(field0: messages));
  }

  updateMessageBox(ChatResponse response) async {
    final box = state.messageBox
        .where((element) =>
            element is ResponseMessageBox && element.id == response.uuid)
        .firstOrNull;

    if (box != null) {
      final l = List<MessageBox>.from(state.messageBox)..remove(box);
      box.content += response.content ?? "";
      if (box is ResponseMessageBox) {
        box.stage = response.stage ?? "";
        box.tokenGenetated = response.tokenGenerated ?? 0;
        box.tps = response.tps ?? 0;
      }
      // state = MessageState(
      //   messageBox: l..add(box),
      //   isLoading: !(response.done ?? false),
      // );
      state = state.copyWith(
        messageBox: l..add(box),
        isLoading: !(response.done ?? false),
      );

      if (response.done == true) {
        await saveHistory(box.content, ChatRole.assistant);
      }
    } else {
      final l = List<MessageBox>.from(state.messageBox)
        ..add(ResponseMessageBox(
            content: response.content ?? "",
            id: response.uuid!,
            stage: response.stage ?? ""));
      // state = MessageState(
      //   isLoading: !(response.done ?? false),
      //   messageBox: l,
      // );
      state = state.copyWith(
        isLoading: !(response.done ?? false),
        messageBox: l,
      );
    }

    scrollController.jumpTo(
      scrollController.position.maxScrollExtent,
    );
  }

  setLoading(bool b) {
    if (b != state.isLoading) {
      state = state.copyWith(isLoading: b);
    }
  }

  void updateUseThoughtChain(bool value) {
    state = state.copyWith(useThoughtChain: value);
  }

  void updateWithHistory(bool value) {
    state = state.copyWith(useHistory: value);
  }
}

final messageProvider =
    AutoDisposeNotifierProvider<MessageNotifier, MessageState>(
  () => MessageNotifier(),
);
