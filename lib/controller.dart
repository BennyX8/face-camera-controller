import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class FaceCameraController extends CameraController {
  FaceCameraController({
    required CameraDescription description,
    ResolutionPreset resolutionPreset = ResolutionPreset.high,
    super.enableAudio = false,
    super.imageFormatGroup,
  }) : super(description, resolutionPreset) {
    _inputImageController = StreamController.broadcast();

    final options = FaceDetectorOptions();
    _detector = FaceDetector(options: options);
  }

  late final StreamController<List<Face>> _inputImageController;

  late final FaceDetector _detector;

  Stream<FaceState> get stream =>
      _inputImageController.stream.transform(StreamTransformer.fromHandlers(
        handleData: (faces, sink) {
          sink.add(switch (faces.length) {
            0 => const EmptyFaceState(),
            1 => SingleFaceState.fromFace(faces.first),
            _ => MultiFaceState(
                count: faces.length,
                faces: [...faces.map(SingleFaceState.fromFace)],
              ),
          });
        },
      ));

  var isChecking = false;

  @override
  Future<void> initialize({
    bool initializeProccessImage = true,
    Duration delay = const Duration(milliseconds: 100),
  }) async {
    await super.initialize();

    if (!initializeProccessImage) {
      return;
    }

    await startImageStream((image) async {
      if (!isChecking) {
        isChecking = true;
        final visionImage = _inputImageFromCameraImage(image);

        if (visionImage == null) {
          isChecking = false;
          return;
        }

        try {
          final faces =
              await Future.microtask(() => _detector.processImage(visionImage));
          _inputImageController.add(faces);
          await Future.delayed(delay);
        } finally {
          isChecking = false;
        }
      }
    });
  }

  InputImage? _inputImageFromCameraImage(CameraImage image) {
    final rotation =
        InputImageRotationValue.fromRawValue(description.sensorOrientation);
    if (rotation == null) return null;

    final format = InputImageFormatValue.fromRawValue(image.format.raw);
    if (format == null && Platform.isIOS) return null;

    final plane = [...image.planes.expand((element) => element.bytes)];

    return InputImage.fromBytes(
      bytes: Uint8List.fromList(plane),
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: format ?? InputImageFormat.yuv_420_888,
        bytesPerRow: image.planes.first.bytesPerRow,
      ),
    );
  }

  @override
  Future<void> dispose() async {
    await _detector.close();
    await _inputImageController.close();

    await super.dispose();
  }
}

sealed class FaceState {
  final int count;

  const FaceState({
    required this.count,
  });

  @override
  String toString() {
    return switch (this) {
      EmptyFaceState() => 'EmptyFaceState',
      SingleFaceState(:final x, :final y, :final z, :final isFaceStaight) =>
        'SingleFaceState x:$x y:$y z:$z isFaceStaight:$isFaceStaight',
      MultiFaceState(:final count) => 'MultiFaceState $count',
    };
  }
}

class EmptyFaceState extends FaceState {
  const EmptyFaceState() : super(count: 0);
}

class SingleFaceState extends FaceState {
  SingleFaceState({
    required this.x,
    required this.y,
    required this.z,
    required this.boundingBox,
  }) : super(count: 1);

  factory SingleFaceState.fromFace(Face face) {
    return SingleFaceState(
      x: face.headEulerAngleX,
      y: face.headEulerAngleY,
      z: face.headEulerAngleZ,
      boundingBox: face.boundingBox,
    );
  }

  final double? x;
  final double? y;
  final double? z;
  final Rect boundingBox;

  bool get isFaceStaight {
    if (x != null &&
        y != null &&
        z != null &&
        x!.abs() < 15 &&
        y!.abs() < 10 &&
        z!.abs() < 5) {
      return true;
    }

    return false;
  }
}

class MultiFaceState extends FaceState {
  MultiFaceState({
    required super.count,
    required this.faces,
  });

  final List<SingleFaceState> faces;
}
