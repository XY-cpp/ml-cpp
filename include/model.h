#ifndef MODEL_H_
#define MODEL_H_

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <objc/NSObjCRuntime.h>

NS_ASSUME_NONNULL_BEGIN

__attribute__((visibility("hidden")))
@interface ModelInput : NSObject<MLFeatureProvider>

@property(readwrite, nonatomic) CVPixelBufferRef image;
@property(readwrite, nonatomic, assign) NSString *inputName;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithImage:(CVPixelBufferRef)image NS_DESIGNATED_INITIALIZER;
@end

__attribute__((visibility("hidden")))
@interface ModelOutput : NSObject<MLFeatureProvider>

@property(readwrite, nonatomic, strong) MLMultiArray *out;
@property(readwrite, nonatomic, assign) NSString *outputName;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithOut:(MLMultiArray *)out NS_DESIGNATED_INITIALIZER;
@end

__attribute__((visibility("hidden")))
@interface Model : NSObject
@property(readonly, nonatomic, nullable) MLModel *model;
@property(readwrite, nonatomic, assign) NSString *inputName;
@property(readwrite, nonatomic, assign) NSString *outputName;
@property(readwrite, nonatomic) NSInteger inputW;
@property(readwrite, nonatomic) NSInteger inputH;
@property(readwrite, nonatomic) NSInteger outputSize;

- (instancetype)initWithMLModel:(NSURL *)modelURL error:(NSError *_Nullable __autoreleasing *_Nullable)error;

- (nullable ModelOutput *)predictionFromImage:(CVPixelBufferRef)image
                                        error:(NSError *_Nullable __autoreleasing *_Nullable)error;
@end

NS_ASSUME_NONNULL_END

#endif  //MODEL_H_