if(    EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/regression/CMakeLists.txt
   AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/verification/CMakeLists.txt)
  ADD_SUBDIRECTORY(regression)
  message("-- Added Plato regression tests")

  ADD_SUBDIRECTORY(verification)
  message("-- Added Plato verification tests")

else()
  message("-- Plato tests not added. skipping.")
  message("--   To include Plato tests:")
  message("--    git submodule init")
  message("--    git submodule update")
  message("--   Then rerun cmake.")
endif()
