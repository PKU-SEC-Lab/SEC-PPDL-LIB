add_library(Datatype INTERFACE)
target_include_directories(Datatype INTERFACE include)
target_link_libraries(Datatype INTERFACE SEAL::seal)
