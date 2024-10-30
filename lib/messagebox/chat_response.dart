import 'package:ai_tools/src/rust/llm.dart' as rust;

class ChatResponse {
  String? stage;
  String? content;
  String? uuid;
  bool? done;

  ChatResponse({this.stage, this.content, this.uuid, this.done});

  // 从 Map 转换为 ChatResponse 对象
  factory ChatResponse.fromJson(Map<String, dynamic> json) {
    return ChatResponse(
      stage: json['stage'] as String?,
      content: json['content'] as String?,
      uuid: json['uuid'] as String?,
      done: json['done'] as bool?,
    );
  }

  factory ChatResponse.fromRustChatResponse(rust.ChatResponse response) {
    return ChatResponse(
      stage: response.stage,
      content: response.content,
      uuid: response.uuid,
      done: response.done,
    );
  }

  // 将 ChatResponse 对象转换为 Map
  Map<String, dynamic> toJson() {
    return {
      'stage': stage,
      'content': content,
      'uuid': uuid,
      'done': done,
    };
  }

  @override
  String toString() {
    return 'ChatResponse(stage: $stage, content: $content, uuid: $uuid, done: $done)';
  }
}
