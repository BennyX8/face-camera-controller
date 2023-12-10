import 'dart:async';

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

  late final Timer _timer;
  late final FaceDetector _detector;

  Stream<FaceState> get stream =>
      _inputImageController.stream.map((faces) => switch (faces.length) {
            0 => const EmptyFaceState(),
            1 => SingleFaceState.fromFace(faces.first),
            _ => MultiFaceState(
                count: faces.length,
                faces: [...faces.map(SingleFaceState.fromFace)],
              ),
          });

  var isChecking = false;

  @override
  Future<void> initialize() async {
    await super.initialize();

    await startImageStream((image) async {
      if (!isChecking) {
        isChecking = true;
        final visionImage = _visionImage(image, description);

        final faces =
            await Future.microtask(() => _detector.processImage(visionImage));
        _inputImageController.add(faces);
        await Future.delayed(const Duration(milliseconds: 100));
        isChecking = false;
      }
    });
  }

  InputImage _visionImage(CameraImage image, CameraDescription description) {
    final allBytes = WriteBuffer();

    for (final plane in image.planes) {
      allBytes.putUint8List(plane.bytes);
    }

    final plane = image.planes.first;

    final bytes = plane.bytes;

    final imageSize = Size(image.width.toDouble(), image.height.toDouble());

    final imageRotation =
        InputImageRotationValue.fromRawValue(description.sensorOrientation) ??
            InputImageRotation.rotation0deg;

    final inputImageFormat =
        InputImageFormatValue.fromRawValue(image.format.raw) ??
            InputImageFormat.nv21;

    final visionImage = InputImage.fromBytes(
      bytes: bytes,
      metadata: InputImageMetadata(
        bytesPerRow: plane.bytesPerRow,
        size: imageSize,
        rotation: imageRotation,
        format: inputImageFormat,
      ),
    );

    return visionImage;
  }

  @override
  Future<void> dispose() async {
    await _detector.close();
    await _inputImageController.close();

    _timer.cancel();
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
  }) : super(count: 1);

  factory SingleFaceState.fromFace(Face face) {
    return SingleFaceState(
      x: face.headEulerAngleX,
      y: face.headEulerAngleY,
      z: face.headEulerAngleZ,
    );
  }

  final double? x;
  final double? y;
  final double? z;

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
