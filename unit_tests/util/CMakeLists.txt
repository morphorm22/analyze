# Test utility helper library

set( TEST_UTIL_SRC 
     PlatoTestHelpers.cpp 
     PlatoMathHelpers.cpp )
set( TEST_UTIL_HDR 
     PlatoTestHelpers.hpp 
     PlatoMathHelpers.hpp )

add_library( Analyze_UnitTestUtils ${TEST_UTIL_SRC} )

set_target_properties( Analyze_UnitTestUtils PROPERTIES PUBLIC_HEADER "${TEST_UTIL_HDR}" )

target_include_directories( Analyze_UnitTestUtils PRIVATE . )
target_include_directories( Analyze_UnitTestUtils PRIVATE ${PROJECT_SOURCE_DIR}/src )
target_include_directories( Analyze_UnitTestUtils PRIVATE "${PLATOENGINE_PREFIX}/include" )
target_include_directories( Analyze_UnitTestUtils PRIVATE ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} )
target_include_directories( Analyze_UnitTestUtils PRIVATE "${OMEGA_H_PREFIX}/include" )

if( CMAKE_INSTALL_PREFIX )
    install( TARGETS Analyze_UnitTestUtils DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )
endif()

set( TEST_UTIL_LINK_LIBS
     analyzelib
     ${PLATO_LIBS}
     ${Trilinos_LIBRARIES}
     ${Trilinos_TPL_LIBRARIES}
     ${Trilinos_EXTRA_LD_FLAGS}
     ${SOLVER_INTERFACE_LIBRARIES} )

target_link_libraries( Analyze_UnitTestUtils PUBLIC ${TEST_UTIL_LINK_LIBS} )