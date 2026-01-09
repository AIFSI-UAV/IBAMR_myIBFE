## ---------------------------------------------------------------------
##
## Copyright (c) 2020 - 2020 by the IBAMR developers
## All rights reserved.
##
## This file is part of IBAMR.
##
## IBAMR is free software and is distributed under the 3-clause BSD
## license. The full text of the license can be found in the file
## COPYRIGHT at the top level directory of IBAMR.
##
## ---------------------------------------------------------------------

# 官方叫 "IBFE-ex4"，输出名 main2d
add_executable(IBFE-ex4-2d example.cpp)

target_link_libraries(IBFE-ex4-2d PRIVATE IBAMR::IBAMR2d)

set_target_properties(IBFE-ex4-2d PROPERTIES
    OUTPUT_NAME "main2d"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/IBFE/ex4"
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/input2d
    ${CMAKE_BINARY_DIR}/examples/IBFE/ex4/input2d
    COPYONLY
)

add_executable(IBFE-ex4-3d example.cpp)

target_link_libraries(IBFE-ex4-3d PRIVATE IBAMR::IBAMR3d)

set_target_properties(IBFE-ex4-3d PROPERTIES
    OUTPUT_NAME "main3d"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/IBFE/ex4"
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/input2d
    ${CMAKE_BINARY_DIR}/examples/IBFE/ex4/input2d
    COPYONLY
)

add_executable(IBFE-ex4-convergence-2d convergence_tester.cpp)
target_link_libraries(IBFE-ex4-convergence-2d PRIVATE IBAMR::IBAMR2d)
set_target_properties(IBFE-ex4-convergence-2d PROPERTIES
    OUTPUT_NAME "convergence_tester2d"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/IBFE/ex4"
)

add_executable(IBFE-ex4-convergence-3d convergence_tester.cpp)
target_link_libraries(IBFE-ex4-convergence-3d PRIVATE IBAMR::IBAMR3d)
set_target_properties(IBFE-ex4-convergence-3d PROPERTIES
    OUTPUT_NAME "IBFE-ex4-convergence-3d"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/IBFE/ex4"
)

