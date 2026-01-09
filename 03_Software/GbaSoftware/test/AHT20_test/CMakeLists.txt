##############################################################################
# Copyright (C) 2022 Mono Wireless Inc. All Rights Reserved.
# Released under MW-SLA-*J,*E (MONO WIRELESS SOFTWARE LICENSE
# AGREEMENT).
##############################################################################
# CMake definitions.
#
# Note: This definition is intended primarily for editing code in the CLion.
#       Although we have defined the build process from the command line,
#       we have not verified the correctness of the build commands.
# 備考: この定義は CLion での編集を意図して作成しています。コマンドラインから
#       ビルドできるように定義を整備していますが、ビルドコマンド等の正確性を
#       保証するものではありません。
# 
# Example: (using with Ninja generator)
# 1. call cmake command
#
#     $ cmake -G Ninja -B build/objs_BLUE_Ninja -DTWELITE=BLUE -DCMAKE_BUILD_TYPE=MinSizeRel
#
#   where 
#     -B specifies the location of Interim Generated Files, etc.
#     -DTWELITE=BLUE specifies building for TWELITE BLUE. (or RED, GOLD)
#     -DCMAKE_BUILD_TYPE specifies optimized option (normal release build should be MinSizeRel)
#         
# 2. build with ninja command.
#   
#     $ ninja -C build/objs_BLUE_Ninja
#
##############################################################################

##############################################################################
# DO NOT EDIT - COMMON DEFINITIONS AND LOADING BODY RULES
cmake_minimum_required(VERSION 3.22)
set(MWS_PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(MWS_PROJECT_NAME ${MWS_PROJECT_ROOT} NAME)
if((NOT $ENV{MWSDK_ROOT} STREQUAL "") AND (IS_DIRECTORY $ENV{MWSDK_ROOT}/TWENET))
    set(MWS_MWSDK $ENV{MWSDK_ROOT})
else()
    get_filename_component(MWS_RELDIR ../.. ABSOLUTE)
    if(IS_DIRECTORY "${MWS_RELDIR}/TWENET")
        set(MWS_MWSDK ${MWS_RELDIR})
    elseif(EXISTS ${MWS_RELDIR}/TWENET.cmake)
        get_filename_component(MWS_MWSDK ${MWS_RELDIR}/../.. ABSOLUTE)
    else()
        message(FATAL_ERROR "!! MWSDK_ROOT cannot be identified.")
    endif()
endif()
######################################################################## END #

##############################################################################
# PROJECT DEPENDENT PART
### INCLUDE DIRS
list(APPEND MWS_INCLUDE_DIRS
        ./
		)

### C SOURCE FILES
file(GLOB MWS_C_SOURCES
        ./*.c
        )

### C++ SOURCE FILES
file(GLOB MWS_CXX_SOURCES
        ./*.cpp
        )

### ADDITIONAL LIBRARIES TO BE LINKED
list(APPEND MWS_LIBS "") # e.g. "-lm"
list(APPEND MWS_LIB_FILES "") # e.g. "./lib/mylib.a"
##############################################################################

##############################################################################
#  DO NOT EDIT - LOAD RULES AND DEFINE PROJECT
include(${MWS_MWSDK}/TWENET/usever.cmake)
project(${MWS_PROJECT_NAME})
######################################################################## END #
