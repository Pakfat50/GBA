#!/bin/bash

######################################################
# THIS IS AN EXPERIMENTAL SCRIPT USING cmake and Ninja
# - ./build-cmake
#       create build rules and build all.
# - ./build-cmake TWELITE={BLUE|RED|GOLD} [clean]
#       for chosen model.
# - ./buuld-cmake cleanall
#       clean all and rebuild.
#
# - In order to update rules in cmake_??? dir,
#   remove cmake_??? dir and re-run this script.
#
######################################################

T=$1
TWELITE="BLUE RED GOLD"
PJROOT=..

if [ "$T" = "cleanall" ]; then
  rm -rfv cmake_*/
  echo
  echo "removed cmake_??? dirs. type $0 to rebuild."
  exit 1
fi
if [ "${T%=*}" = "TWELITE" ]; then
  eval TWELITE=\"${T#TWELITE=}\"
  shift
fi
echo $TWELITE

for f in $TWELITE; do
  echo "--- building cmake_$f ---"
  if [ ! -d cmake_$f ]; then
    cmake -G"Ninja" -DTWELITE=$f -DCMAKE_BUILD_TYPE=MinSizeRel -B cmake_$f -S $PJROOT
  fi
  ninja -C cmake_$f $*
  echo ""
done
