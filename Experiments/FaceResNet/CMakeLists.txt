add_library(FaceResNet STATIC src/FaceResNet.cpp)
target_include_directories(FaceResNet PUBLIC include)
target_link_libraries(FaceResNet PUBLIC Runtime)

add_executable(test_face_resnet src/TestFaceResNet.cpp)
target_link_libraries(test_face_resnet PRIVATE FaceResNet)

add_executable(face_w4a8 EXCLUDE_FROM_ALL w4a8/Compact.cpp)
target_link_libraries(face_w4a8 PRIVATE OptiPrime FaceResNet)
