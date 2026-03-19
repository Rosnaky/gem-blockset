file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "*.cu")

include(GoogleTest)

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    add_executable(${TEST_NAME} ${TEST_SOURCE})

    target_include_directories(${TEST_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/models/primitives
        ${PROJECT_SOURCE_DIR}/models/composites
    )

    target_link_libraries(${TEST_NAME}
        GTest::gtest_main
    )

    gtest_discover_tests(${TEST_NAME})
endforeach()
