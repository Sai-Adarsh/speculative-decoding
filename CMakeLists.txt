cmake_minimum_required(VERSION 3.20)
project(speculative_decoding LANGUAGES CXX)

add_executable(speculative-decoding main.cpp)
target_compile_features(speculative-decoding PRIVATE cxx_std_20)
target_compile_options(speculative-decoding PRIVATE -Wall -Wextra -Wpedantic)

option(ENABLE_LLAMA_SERVER "Build the optional llama.cpp HTTP client" OFF)
if(ENABLE_LLAMA_SERVER)
  find_package(CURL REQUIRED)
  target_sources(speculative-decoding PRIVATE llama_client.cpp)
  target_link_libraries(speculative-decoding PRIVATE CURL::libcurl)
  target_compile_definitions(speculative-decoding PRIVATE ENABLE_LLAMA_SERVER)
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(NAME speculative_fixture_cli
    COMMAND "${CMAKE_COMMAND}"
      "-DSPEC=$<TARGET_FILE:speculative-decoding>"
      "-DSERVER_ENABLED=${ENABLE_LLAMA_SERVER}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixture.cmake")
  if(ENABLE_LLAMA_SERVER)
    add_executable(speculative_loopback_test tests/loopback.cpp)
    target_compile_features(speculative_loopback_test PRIVATE cxx_std_20)
    add_test(NAME speculative_server_cli
      COMMAND speculative_loopback_test "$<TARGET_FILE:speculative-decoding>")
    set_tests_properties(speculative_server_cli PROPERTIES TIMEOUT 40)
  endif()
endif()
