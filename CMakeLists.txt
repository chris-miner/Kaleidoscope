cmake_minimum_required(VERSION 3.0.0)
project(Kaleidoscope VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(Kaleidoscope main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

target_include_directories(Kaleidoscope PUBLIC /usr/local/opt/llvm/include)
target_link_libraries(Kaleidoscope /usr/local/opt/llvm/lib/libLLVM.dylib)