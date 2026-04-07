@ECHO OFF

set SPHINXBUILD=sphinx-build
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "clean" (
    rmdir /s /q %BUILDDIR%
    goto end
)

%SPHINXBUILD% -W -b html %SOURCEDIR% %BUILDDIR%/html

:end
