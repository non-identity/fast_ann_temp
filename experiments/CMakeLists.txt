project(FastANN)

set(DATASET_NAME siftsmall)
set(DATASET_FOLDER ${CMAKE_BINARY_DIR}/datasets/${DATASET_NAME})
set(DATASET_FILE ${DATASET_FOLDER}.tar.gz)
set(DATASET_MD5 0b8324a7a82d7f2663d7dcbd57642df7)
set(DATASET_URL ftp://ftp.irisa.fr/local/texmex/corpus/${DATASET_NAME}.tar.gz)

if(EXISTS ${DATASET_FOLDER})
    message(STATUS "Dataset is already downloaded... skipping download.")
else()
    file(DOWNLOAD   ${DATASET_URL}
                    ${DATASET_FILE}
                    SHOW_PROGRESS
                    EXPECTED_MD5 ${DATASET_MD5})
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${DATASET_NAME}.tar.gz
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/datasets)
    file(REMOVE ${DATASET_FILE})
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(GLOB RUNNER_SOURCES RELATIVE_PATH *.cpp)
foreach(runner_source ${RUNNER_SOURCES})
    get_filename_component(runner_bin ${runner_source} NAME_WE)
    add_executable(${runner_bin} ${runner_source})
    target_link_libraries(${runner_bin} fast_ann)
endforeach(runner_source ${RUNNER_SOURCES})

file(COPY ${CMAKE_SOURCE_DIR}/scripts DESTINATION ${CMAKE_BINARY_DIR})
