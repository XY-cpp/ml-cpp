#include <Foundation/Foundation.h>
#include <os/object.h>

#import "model.h"

@implementation ModelInput

- (instancetype)initWithImage:(CVPixelBufferRef)image {
  self = [super init];
  if (self != nullptr) {
    _image = image;
    CVPixelBufferRetain(_image);
  }
  return self;
}

- (void)dealloc {
  CVPixelBufferRelease(_image);
  [super dealloc];
}

- (NSSet<NSString *> *)featureNames {
  return [NSSet setWithArray:@[ _inputName ]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
  if ([featureName isEqualToString:_inputName] != 0) {
    return [MLFeatureValue featureValueWithPixelBuffer:self.image];
  }
  return nil;
}

@end

@implementation ModelOutput

- (instancetype)initWithOut:(MLMultiArray *)out {
  self = [super init];
  if (self != nullptr) {
    _out = out;
  }
  return self;
}

- (NSSet<NSString *> *)featureNames {
  return [NSSet setWithArray:@[ _outputName ]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
  if ([featureName isEqualToString:_outputName] != 0) {
    return [MLFeatureValue featureValueWithMultiArray:self.out];
  }
  return nil;
}

- (void)dealloc {
  [_out dealloc];
  [super dealloc];
}

@end

@implementation Model

- (instancetype)initWithMLModel:(NSURL *)modelURL error:(NSError *_Nullable __autoreleasing *_Nullable)error {
  self = [super init];
  if (self == nullptr) {
    return nil;
  }
  MLModel *model = [MLModel modelWithContentsOfURL:modelURL error:error];
  _model = model;
  if (_model == nil) {
    return nil;
  }
  _inputName = model.modelDescription.inputDescriptionsByName.allKeys.firstObject;
  _outputName = model.modelDescription.outputDescriptionsByName.allKeys.firstObject;
  MLModelDescription *model_description = [_model modelDescription];
  NSDictionary<NSString *, MLFeatureDescription *> *input_descriptions = [model_description inputDescriptionsByName];
  NSDictionary<NSString *, MLFeatureDescription *> *output_descriptions = [model_description outputDescriptionsByName];
  _inputH = ((MLImageConstraint *)[input_descriptions[_inputName] imageConstraint]).pixelsHigh;
  _inputW = ((MLImageConstraint *)[input_descriptions[_inputName] imageConstraint]).pixelsWide;
  NSArray<NSNumber *> *output_shape =
      ((MLMultiArrayConstraint *)[output_descriptions[_outputName] multiArrayConstraint]).shape;
  _outputSize = 1;
  for (NSNumber *dimension in output_shape) {
    _outputSize *= [dimension unsignedIntegerValue];
  }
  return self;
}

- (nullable ModelOutput *)predictionFromImage:(CVPixelBufferRef)image
                                        error:(NSError *_Nullable __autoreleasing *_Nullable)error {
  ModelInput *input = [[ModelInput alloc] initWithImage:image];
  input.inputName = _inputName;
  MLPredictionOptions *options = [[MLPredictionOptions alloc] init];
  id<MLFeatureProvider> out_features = [self.model predictionFromFeatures:input options:options error:error];
  if (out_features == nullptr) {
    return nil;
  }
  [input dealloc];
  [options dealloc];
  ModelOutput *output =
      [[ModelOutput alloc] initWithOut:(MLMultiArray *)[out_features featureValueForName:_outputName].multiArrayValue];
  output.outputName = _outputName;
  return output;
}

- (void)dealloc {
  [_model dealloc];
  [super dealloc];
}
@end