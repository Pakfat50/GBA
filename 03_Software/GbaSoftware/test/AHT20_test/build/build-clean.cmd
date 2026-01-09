@echo off

if exist %MWSDK_ROOT_WINNAME%\000manifest GOTO RUN

set CDSAVE=%CD%

cd ..\..\..
set MWSDK_ROOT_WINNAME=%CD%
if exist 000manifest GOTO RUN

cd /D %CDSAVE%
cd ..\..\..\..
set MWSDK_ROOT_WINNAME=%CD%
if exist 000manifest GOTO RUN

echo MWSDK ROOT DIR is not found. Please set MWSDK_ROOT_WINNAME proerly.
exit 1

:RUN
cd /D %CDSAVE%
SET F=%~n0
SET FARG=%F:build-=%

echo %MWSDK_ROOT_WINAME%

if %FARG%==clean (
  CALL %MWSDK_ROOT_WINNAME%\scripts\WIN_BASH.cmd make cleanall
) else (
  CALL %MWSDK_ROOT_WINNAME%\scripts\WIN_BASH.cmd make TWELITE=%FARG% all
)

PAUSE
:_End
