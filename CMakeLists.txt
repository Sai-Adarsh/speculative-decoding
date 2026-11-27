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
