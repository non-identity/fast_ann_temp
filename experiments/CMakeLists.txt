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

add_executable(runner runner.cpp)
target_link_libraries(runner fast_ann)

# find_package (Python3 COMPONENTS Interpreter)
