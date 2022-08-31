###############################################################################
# Sources:
###############################################################################
SET(BamG_SRCS BamG.cpp)

SET(BamG_HDRS BamG.hpp)

add_library(BamGlib ${BamG_SRCS} ${BamG_HDRS})

add_executable(BamG BamG_main.cpp)

set(BamG_LIBS BamGlib ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} )
target_include_directories(BamGlib PRIVATE ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})

target_link_libraries(BamG ${BamG_LIBS})
target_include_directories(BamG PRIVATE ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})

if( CMAKE_INSTALL_PREFIX )
  install( TARGETS BamG DESTINATION ${CMAKE_INSTALL_PREFIX}/bin )
endif()

###############################################################################
###############################################################################

add_subdirectory(./unittest)
