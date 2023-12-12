import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class FaceCameraController extends CameraController {
  FaceCameraController({
    required CameraDescription description,
    this.processDelay = const Duration(milliseconds: 100),
    ResolutionPreset resolutionPreset = ResolutionPreset.high,
    FaceDetectorOptions? faceDetectorOptions,
    super.enableAudio = false,
    super.imageFormatGroup,
  }) : super(description, resolutionPreset) {
    _inputImageController = StreamController.broadcast();

    final options = faceDetectorOptions ?? FaceDetectorOptions();
    _detector = FaceDetector(options: options);
  }

  final Duration processDelay;

  late final StreamController<List<Face>> _inputImageController;
  late final FaceDetector _detector;

  bool _isDisposed = false;
  bool _isChecking = false;

  Stream<FaceState> get stream => _inputImageController.stream.transform(
        StreamTransformer.fromHandlers(
          handleData: (faces, sink) {
            sink.add(
              switch (faces.length) {
                0 => const EmptyFaceState(),
                1 => SingleFaceState.fromFace(faces.first),
                _ => MultiFaceState(
                    count: faces.length,
                    faces: [...faces.map(SingleFaceState.fromFace)],
                  ),
              },
            );
          },
        ),
      );

  Future<List<Face>> processImage(InputImage inputImage) async {
    return _detector.processImage(inputImage);
  }

  @override
  Future<void> initialize({
    bool initializeProccessImage = true,
  }) async {
    await super.initialize();

    if (!initializeProccessImage) {
      return;
    }

    await startImageStream(_processCameraImage);
  }

  @override
  Future<void> startVideoRecording({
    onLatestImageAvailable? onAvailable,
  }) async {
    await super.startVideoRecording(
      onAvailable: (image) {
        onAvailable?.call(image);
        _processCameraImage(image);
      },
    );
  }

  @override
  Future<XFile> stopVideoRecording() async {
    final file = await super.stopVideoRecording();

    if (!value.isStreamingImages) {
      await startImageStream(_processCameraImage);
    }

    return file;
  }

  @override
  Future<void> dispose() async {
    await _detector.close();
    await _inputImageController.close();

    _isDisposed = true;
    await super.dispose();
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (!_isChecking) {
      _isChecking = true;
      final visionImage = _inputImageFromCameraImage(image);

      if (visionImage == null) {
        _isChecking = false;
        return;
      }

      try {
        final faces = await Future.microtask(() => processImage(visionImage));

        if (!_isDisposed) {
          _inputImageController.add(faces);
        }

        await Future.delayed(processDelay);
      } finally {
        _isChecking = false;
      }
    }
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
