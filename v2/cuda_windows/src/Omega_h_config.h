#ifndef OMEGA_H_CONFIG_H
#define OMEGA_H_CONFIG_H

#define OMEGA_H_USE_CUDA
#define OMEGA_H_VERSION_MAJOR 9
#define OMEGA_H_VERSION_MINOR 31
#define OMEGA_H_VERSION_PATCH 2
#define OMEGA_H_SEMVER "9.31.2+0001000000000"
#define OMEGA_H_COMMIT ""
#define OMEGA_H_CXX_FLAGS "/DWIN32 /D_WINDOWS /GR /EHsc"
#define OMEGA_H_CMAKE_ARGS "-DBUILD_SHARED_LIBS:BOOL=\"False\" -DOmega_h_USE_ZLIB:BOOL=\"False\" -DOmega_h_USE_OpenMP:BOOL=\"OFF\" -DOmega_h_USE_CUDA:BOOL=\"True\""

#endif
