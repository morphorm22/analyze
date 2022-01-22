#  run verification problem
set( TEST_COMMAND "${ANALYZE_EXECUTABLE} --output-viz=${OUTPUT_DATA} --input-config=${CONFIG_FILE}" )
execute_process(COMMAND bash "-c" "${TEST_COMMAND}" RESULT_VARIABLE HAD_ERROR)
if (HAD_ERROR)
  message(FATAL_ERROR "FAILED: ${TEST_COMMAND}")
endif()


#  generate comparison
set( TEST_COMMAND "${DIFF} -f ${DIFF_CONF} ${OUTPUT_DATA} ${GOLD_DATA}" )
execute_process(COMMAND bash "-c" "${TEST_COMMAND}" RESULT_VARIABLE HAD_ERROR)
if (HAD_ERROR)
  message(FATAL_ERROR "FAILED: ${TEST_COMMAND}")
endif()
