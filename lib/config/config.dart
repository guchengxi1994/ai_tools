import 'package:json_annotation/json_annotation.dart';

part 'config.g.dart';

@JsonSerializable()
class ServerConfig {
  final Server server;
  final List<Model> models;

  ServerConfig({required this.server, required this.models});

  factory ServerConfig.fromJson(Map<String, dynamic> json) =>
      _$ServerConfigFromJson(json);
  Map<String, dynamic> toJson() => _$ServerConfigToJson(this);
}

@JsonSerializable()
class Server {
  final int port;

  Server({required this.port});

  factory Server.fromJson(Map<String, dynamic> json) => _$ServerFromJson(json);
  Map<String, dynamic> toJson() => _$ServerToJson(this);
}

@JsonSerializable()
class Model {
  final String name;
  final String type;
  final List<String> catalogs;

  @JsonKey(name: 'available-download-url')
  final String availableDownloadUrl;

  @JsonKey(name: 'local-path')
  final String localPath;

  Model({
    required this.name,
    required this.type,
    required this.catalogs,
    required this.availableDownloadUrl,
    required this.localPath,
  });

  factory Model.fromJson(Map<String, dynamic> json) => _$ModelFromJson(json);
  Map<String, dynamic> toJson() => _$ModelToJson(this);
}
