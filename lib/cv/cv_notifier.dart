import 'package:ai_tools/src/rust/api/cv.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

enum CvModels { yolov8, efficientnet, beit, none }

enum CvTask { objectDetect, imageClassification, none }

extension CvModelsExtension on CvModels {
  String get name {
    switch (this) {
      case CvModels.yolov8:
        return "Yolov8";
      case CvModels.efficientnet:
        return "EfficientNet";
      case CvModels.beit:
        return "Beit";
      default:
        return "None";
    }
  }

  Future onSelect(WidgetRef ref) async {
    ref.read(cvStateProvider.notifier).changeLoading(this);
    switch (this) {
      case CvModels.yolov8:
        await loadYolov8(
            modelPath:
                r"D:\github_repo\ai_tools\rust\assets\yolov8n.safetensors");
        ref.read(cvStateProvider.notifier).changeModel(this);
        ref.read(cvStateProvider.notifier).changeLoading(CvModels.none);
        break;
      case CvModels.efficientnet:
        await loadEfficientnet(
            modelPath:
                r"D:\github_repo\ai_tools\rust\assets\efficientnet-lite4-s.onnx");
        ref.read(cvStateProvider.notifier).changeModel(this);
        ref.read(cvStateProvider.notifier).changeLoading(CvModels.none);
        break;
      case CvModels.beit:
        await loadBeit(
            modelPath:
                r"D:\github_repo\ai_tools\rust\assets\beit_base_patch16_384.in22k_ft_in22k_in1k.safetensors");
        ref.read(cvStateProvider.notifier).changeModel(this);
        ref.read(cvStateProvider.notifier).changeLoading(CvModels.none);
        break;
      case CvModels.none:
        ref.read(cvStateProvider.notifier).changeModel(this);
        ref.read(cvStateProvider.notifier).changeLoading(CvModels.none);
    }
  }

  CvTask get task {
    switch (this) {
      case CvModels.yolov8:
        return CvTask.objectDetect;
      case CvModels.efficientnet:
        return CvTask.imageClassification;
      case CvModels.beit:
        return CvTask.imageClassification;
      default:
        return CvTask.none;
    }
  }
}

class CvState {
  final CvModels activeModel;
  final CvModels? loadingModel;
  late CvTask task;
  final bool isExpanded;

  CvState(
      {this.activeModel = CvModels.none,
      this.loadingModel,
      this.isExpanded = false}) {
    task = activeModel.task;
  }

  CvState copyWith({
    CvModels? activeModel,
    CvModels? loadingModel,
    bool? isExpanded,
  }) {
    return CvState(
      activeModel: activeModel ?? this.activeModel,
      loadingModel: loadingModel ?? this.loadingModel,
      isExpanded: isExpanded ?? this.isExpanded,
    );
  }
}

class CvNotifier extends Notifier<CvState> {
  @override
  CvState build() {
    return CvState();
  }

  void changeLoading(CvModels model) {
    if (null != state.loadingModel) return;
    state = CvState(
        activeModel: state.activeModel,
        loadingModel: model,
        isExpanded: state.isExpanded);
  }

  void changeModel(CvModels model) {
    if (model == state.activeModel) return;
    state = CvState(
        activeModel: model, loadingModel: null, isExpanded: state.isExpanded);
  }

  changeExpand() {
    state = state.copyWith(isExpanded: !state.isExpanded);
  }
}

final cvStateProvider = NotifierProvider<CvNotifier, CvState>(CvNotifier.new);
