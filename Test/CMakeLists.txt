if(USE_HE_GPU)
    add_executable(testHEGPU ${CMAKE_CURRENT_LIST_DIR}/src/TestHEGPU.cpp)
    target_link_libraries(testHEGPU PUBLIC HE Datatype)
endif()

add_executable(test_linear_operator ${CMAKE_CURRENT_LIST_DIR}/src/TestLinearOperator.cpp)
target_link_libraries(test_linear_operator PUBLIC LinearOperator Utils)

add_executable(test_Conv ${CMAKE_CURRENT_LIST_DIR}/src/TestConv.cpp)
target_link_libraries(test_Conv PRIVATE LinearLayer)

add_executable(test_matmul ${CMAKE_CURRENT_LIST_DIR}/src/TestMatmul.cpp)
target_link_libraries(test_matmul PRIVATE LinearLayer)

add_executable(test_cir_linear ${CMAKE_CURRENT_LIST_DIR}/src/TestCirLinear.cpp)
target_link_libraries(test_cir_linear PRIVATE LinearLayer)

add_executable(test_cir_conv ${CMAKE_CURRENT_LIST_DIR}/src/TestCirConv.cpp)
target_link_libraries(test_cir_conv PRIVATE LinearLayer)

add_executable(test_ntt ${CMAKE_CURRENT_LIST_DIR}/src/TestNTT.cpp)
target_link_libraries(test_ntt PUBLIC Utils)

add_executable(test_fixpoint ${CMAKE_CURRENT_LIST_DIR}/src/TestFixPoint.cpp)
target_link_libraries(test_fixpoint PUBLIC NonlinearOperator)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_executable(test_optiencode_conv src/TestOptiEncodeConv.cpp)
target_link_libraries(test_optiencode_conv PRIVATE LinearLayer)

function(add_optiencode_test name target)
    add_test(NAME ${name} COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_optiencode.py
        --binary $<TARGET_FILE:${target}> --logs ${CMAKE_BINARY_DIR}/test-results/${name} ${ARGN})
    set_tests_properties(${name} PROPERTIES TIMEOUT 650 RUN_SERIAL TRUE)
endfunction()
add_test(NAME optiencode_algebra COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/verify_optiencode.py)
add_optiencode_test(optiencode_conv test_optiencode_conv)
add_optiencode_test(optiencode_legacy_keys test_optiencode_conv --default-keys 1 --threads 3 --repeat 1)
if(TARGET test_face_resnet)
    add_optiencode_test(optiencode_face test_face_resnet --face --threads 3 --timeout 600)
endif()
add_optiencode_test(cheetah_conv_reference test_optiencode_conv --bench 1 --threads 2 --repeat 1)
if(USE_HE_GPU)
    add_optiencode_test(optiencode_gpu_conv test_optiencode_conv --gpu 1 --custom 1 --threads 4)
endif()

add_executable(test_tensor_storage src/TestTensorStorage.cpp)
target_link_libraries(test_tensor_storage PRIVATE Datatype)
add_test(NAME tensor_storage COMMAND test_tensor_storage)
