// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'config.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

ServerConfig _$ServerConfigFromJson(Map<String, dynamic> json) => ServerConfig(
      server: Server.fromJson(json['server'] as Map<String, dynamic>),
      models: (json['models'] as List<dynamic>)
          .map((e) => Model.fromJson(e as Map<String, dynamic>))
          .toList(),
    );

Map<String, dynamic> _$ServerConfigToJson(ServerConfig instance) =>
    <String, dynamic>{
      'server': instance.server,
      'models': instance.models,
    };

Server _$ServerFromJson(Map<String, dynamic> json) => Server(
      port: (json['port'] as num).toInt(),
    );

Map<String, dynamic> _$ServerToJson(Server instance) => <String, dynamic>{
      'port': instance.port,
    };

Model _$ModelFromJson(Map<String, dynamic> json) => Model(
      name: json['name'] as String,
      type: json['type'] as String,
      catalogs:
          (json['catalogs'] as List<dynamic>).map((e) => e as String).toList(),
      availableDownloadUrl: json['available-download-url'] as String,
      localPath: json['local-path'] as String,
    );

Map<String, dynamic> _$ModelToJson(Model instance) => <String, dynamic>{
      'name': instance.name,
      'type': instance.type,
      'catalogs': instance.catalogs,
      'available-download-url': instance.availableDownloadUrl,
      'local-path': instance.localPath,
    };
